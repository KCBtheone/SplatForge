"""Train planar Gaussians and extract a surface from COLMAP inputs."""
from __future__ import annotations

import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

PGSR_REQUIRED_FILES = (
    "tools/commercial_colmap.py",
    "unbiased_gsplat/__init__.py",
    "unbiased_gsplat/commercial_model.py",
    "unbiased_gsplat/gradient_validation.py",
    "unbiased_gsplat/multiview.py",
    "unbiased_gsplat/photometric.py",
    "unbiased_gsplat/planar_gaussian.py",
    "unbiased_gsplat/refinement_schedule.py",
    "unbiased_gsplat/view_tensor_cache.py",
)


def source_root(source: Path | None = None) -> Path:
    configured = source if source is not None else os.environ.get("PGSR_SOURCE_ROOT")
    if not configured:
        raise RuntimeError("Set PGSR_SOURCE_ROOT to the standalone PGSR source directory")
    root = Path(configured).expanduser().resolve()
    missing = [name for name in PGSR_REQUIRED_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete PGSR_SOURCE_ROOT {root}. Missing: {', '.join(missing)}")
    for path in (root, root / "tools"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return root


def camera_path_diagnostics(rows) -> dict:
    """Check the upright capture assumption before learning scene geometry."""
    import numpy as np
    rotations = np.stack([row["c2w"][:3, :3] for row in rows])
    ups = -rotations[:, :, 1]
    up = ups.mean(0)
    coherence = float(np.linalg.norm(up))
    if coherence < .1:
        raise ValueError("SfM camera up direction is ambiguous; repair camera inputs before PGSR")
    opposed = np.flatnonzero(ups @ up <= 0)
    relative = np.transpose(rotations[:-1], (0, 2, 1)) @ rotations[1:]
    angles = np.degrees(np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2)-1)/2, -1, 1)))
    result = dict(camera_up_coherence=coherence, opposed_up_views=opposed.tolist(),
                  maximum_adjacent_rotation_degrees=float(angles.max()) if len(angles) else 0.)
    if len(opposed):
        raise ValueError(f"SfM has {len(opposed)}/{len(rows)} reversed camera-up directions "
                         f"in an upright capture (views {opposed.tolist()}); "
                         "repair camera inputs before PGSR")
    return result


def train_and_extract(
    data: Path, output: Path, options: dict, *, source: Path | None = None,
    device="cuda", export_only=False, export_on_finish=True, mesh_path: Path | None = None,
) -> tuple[Path | None, dict]:
    root = source_root(source)
    for module_name in ("unbiased_gsplat", "commercial_colmap"):
        module = sys.modules.get(module_name)
        if module is not None:
            location = getattr(module, "__file__", None)
            if not location or not Path(location).resolve().is_relative_to(root):
                raise RuntimeError(f"{module_name} is already loaded from a different source directory")
    import numpy as np
    import torch
    from splatforge._paths import data_root, output_path
    from commercial_colmap import ColmapInputs
    from unbiased_gsplat.commercial_model import (
        initialize_parameters, make_optimizer, update_mean_learning_rate, refine_population)
    from unbiased_gsplat.planar_gaussian import (
        PlanarGaussianConfig, render_planar_gaussians, edge_aware_normal_loss, plane_depth)
    from unbiased_gsplat.multiview import (
        PGSRMultiViewConfig, build_camera_neighbor_graph,
        multiview_consistency, weighted_multiview_regularization)
    from unbiased_gsplat.photometric import structural_similarity, structural_similarity_target_statistics
    from unbiased_gsplat.gradient_validation import validate_adam_inputs
    from unbiased_gsplat.refinement_schedule import reset_opacities_if_due
    from unbiased_gsplat.view_tensor_cache import ViewTensorCache

    if not torch.cuda.is_available():
        raise RuntimeError("PGSR training requires an NVIDIA CUDA GPU")
    selected = torch.device(device)
    if selected.type != "cuda":
        raise ValueError("PGSR requires a CUDA device")
    if selected.index is not None:
        torch.cuda.set_device(selected)
    output = output_path(output)
    storage = data_root()

    def check_disk():
        if storage is not None and shutil.disk_usage(storage).free < 20 * 2**30:
            raise RuntimeError("The configured data disk has less than 20 GiB free")

    check_disk()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint.pt"
    if export_only and not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    steps, seed = options["steps"], options["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    inputs = ColmapInputs(data, factor=options["factor"], cache_views=12)
    camera_quality = camera_path_diagnostics(inputs.rows)
    keep, _, _ = inputs.point_partition()
    points, colors = inputs.points[keep], inputs.colors[keep]
    if len(points) < 4 or not np.isfinite(points).all() or inputs.radius <= 0:
        raise ValueError("Insufficient finite foreground SfM points for PGSR")
    if options["max_points"] and len(points) > options["max_points"]:
        selected = np.random.default_rng(seed).choice(len(points), options["max_points"], replace=False)
        points, colors = points[selected], colors[selected]

    camera_precompute_seconds = 0.

    def tensors(item, *, calibration=None, training=True):
        nonlocal camera_precompute_seconds
        result = {k: torch.as_tensor(item[k], device="cuda") for k in ("foreground", "K", "c2w")}
        result["rgb"] = torch.as_tensor(item["rgb"], device="cuda", dtype=torch.float32) / 255
        result["rgb"] = result["rgb"] * result["foreground"][..., None]
        if training and options.get("ssim_target_cache", False):
            result["ssim_target_mean"], result["ssim_target_variance"] = structural_similarity_target_statistics(result["rgb"])
        if options.get("camera_geometry_cache", False):
            if calibration is None:
                tick = time.perf_counter()
                k = result["K"]
                if bool((k[0, 0] == 0).item()) or bool((k[1, 1] == 0).item()):
                    raise ValueError("camera focal lengths must be nonzero")
                result["world_to_camera"] = torch.linalg.inv(result["c2w"])
                result["intrinsics_inverse"] = torch.linalg.inv(k)
                camera_precompute_seconds += time.perf_counter() - tick
            else:
                result.update(calibration)
        return result

    plane_depth_function = None

    def camera_arguments(item):
        arguments = {}
        if options.get("camera_geometry_cache", False):
            arguments.update(world_to_camera=item["world_to_camera"],
                             intrinsics_inverse=item["intrinsics_inverse"], focal_lengths_checked=True)
        if plane_depth_function is not None:
            arguments["plane_depth_function"] = plane_depth_function
        return arguments

    def extract(parameters, updates, calibrations):
        import open3d as o3d
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=options["tsdf_voxel"], sdf_trunc=options["tsdf_trunc"],
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        started = time.perf_counter()
        with torch.no_grad():
            for index in range(len(inputs.rows)):
                item = tensors(inputs.load(index), calibration=calibrations.get(index), training=False)
                h, w = item["rgb"].shape[:2]
                layer = render_planar_gaussians(parameters, item["c2w"], item["K"], w, h,
                                                min(3, max(0, (updates - 1) // 1000)), **camera_arguments(item))
                valid = layer["valid"] & item["foreground"] & (layer["alpha"] >= .5)
                depth = torch.where(valid, layer["depth"], 0).cpu().numpy().astype(np.float32)
                rgb = np.uint8(np.clip(inputs.load(index)["rgb"], 0, 255))
                k = item["K"].cpu().numpy()
                intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, float(k[0, 0]), float(k[1, 1]),
                                                            float(k[0, 2]), float(k[1, 2]))
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(rgb.copy()), o3d.geometry.Image(depth), depth_scale=1.,
                    depth_trunc=20., convert_rgb_to_intensity=False)
                volume.integrate(rgbd, intrinsic, np.linalg.inv(item["c2w"].cpu().numpy()))
                del item, layer
        raw = volume.extract_triangle_mesh()
        if not len(raw.triangles):
            raise RuntimeError("PGSR TSDF extraction produced no surface")
        path = mesh_path if mesh_path is not None else output / "raw.ply"
        check_disk()
        temporary = path.with_suffix(".tmp.ply") if mesh_path is not None else path
        if not o3d.io.write_triangle_mesh(str(temporary), raw, write_vertex_normals=False):
            raise RuntimeError("Could not write PGSR mesh")
        if mesh_path is not None:
            temporary.replace(path)
        return path, time.perf_counter() - started

    parent = torch.load(checkpoint_path, map_location="cpu", weights_only=False) if checkpoint_path.exists() else None
    postprocess = {"representation", "hair_box", "tsdf_voxel", "tsdf_trunc", "cut_fraction", "height_mm", "base_margin_mm", "target_faces",
                   "gpu_view_cache", "camera_geometry_cache", "tensor_loss_fusion", "reuse_render_points", "loss_cudagraphs", "ssim_target_cache", "plane_depth_fusion", "gpu_sift_matching", "separable_ssim"}
    if parent is not None:
        if not isinstance(parent, dict) or not {"options", "views", "parameters", "updates"} <= parent.keys():
            raise ValueError("Expected a PGSR checkpoint")
        if any(parent["options"].get(k) != v for k, v in options.items() if k not in postprocess) or parent["views"] != [r["name"] for r in inputs.rows]:
            raise ValueError("Existing checkpoint uses different inputs or parameters")
        parameters = {k: torch.nn.Parameter(v.cuda()) for k, v in parent["parameters"].items()}
    else:
        parameters = initialize_parameters(points, colors, axes=3, seed=seed)
    if export_only:
        path, seconds = extract(parameters, parent["updates"], {})
        return path, dict(representation="pgsr_only", updates=parent["updates"],
                          extraction_seconds=seconds)

    cache = ViewTensorCache(inputs.load, tensors, enabled=options["gpu_view_cache"])
    pg_config = PlanarGaussianConfig(normal_weight=options["normal_weight"])
    mv_config = PGSRMultiViewConfig()
    graph = build_camera_neighbor_graph(
        torch.tensor(np.stack([r["c2w"] for r in inputs.rows]), dtype=torch.float32),
        list(range(len(inputs.rows))), config=mv_config)
    training = set(inputs.train_indices)
    graph = [tuple(i for i in row if i in training) for row in graph]
    families = {"pgsr": parameters}
    exposure = torch.nn.Parameter(torch.zeros(len(inputs.rows), 2, device="cuda"))
    optimizer = make_optimizer(families, exposure)
    sums = torch.zeros(len(parameters["means"]), device="cuda")
    counts = torch.zeros_like(sums)
    stack = []
    updates = 0
    if parent is not None:
        optimizer.load_state_dict(parent["optimizer"])
        with torch.no_grad():
            exposure.copy_(parent["exposure"].cuda())
        updates = parent["updates"]
        sums, counts = parent["sums"].cuda(), parent["counts"].cuda()
        stack = parent["stack"]
        random.setstate(parent["random_state"])
        np.random.set_state(parent["numpy_state"])
        torch.set_rng_state(parent["torch_state"])
        torch.cuda.set_rng_state_all(parent["cuda_state"])

    training_seconds_before = float(parent.get("training_seconds", 0)) if parent else 0.
    multiview_updates = int(parent.get("multiview_updates", 0)) if parent else 0

    def save():
        temporary = checkpoint_path.with_suffix(".tmp")
        torch.save(dict(parameters={k: v.detach().cpu() for k, v in parameters.items()},
                        optimizer=optimizer.state_dict(), exposure=exposure.detach().cpu(),
                        updates=updates, options=options, views=[r["name"] for r in inputs.rows],
                        multiview_updates=multiview_updates,
                        training_seconds=training_seconds_before + time.perf_counter() - started,
                        sums=sums.cpu(), counts=counts.cpu(), stack=stack,
                        random_state=random.getstate(), numpy_state=np.random.get_state(),
                        torch_state=torch.get_rng_state(), cuda_state=torch.cuda.get_rng_state_all()), temporary)
        temporary.replace(checkpoint_path)

    started = time.perf_counter()
    initial_updates = updates
    ssim_forward_calls = 0
    if options.get("separable_ssim", False):
        from functools import partial
        structural_similarity = partial(structural_similarity, separable=True)
    last_disk_check = started
    loss_cudagraphs = options.get("loss_cudagraphs", False)
    if loss_cudagraphs and not hasattr(torch.compiler, "cudagraph_mark_step_begin"):
        raise RuntimeError("loss_cudagraphs requires the production Torch2.7 compiler API")
    if loss_cudagraphs:
        from torch._dynamo.utils import counters as compiler_counters
        from torch._inductor.cudagraph_trees import CUDAGraphNode
        cudagraph_skips_before = int(compiler_counters["inductor"].get("cudagraph_skips", 0))
        original_run_graph = CUDAGraphNode.run_graph
        cudagraph_replays = 0

        def counted_run_graph(node, *args, **kwargs):
            nonlocal cudagraph_replays
            result = original_run_graph(node, *args, **kwargs)
            cudagraph_replays += 1
            return result

        CUDAGraphNode.run_graph = counted_run_graph
    compile_mode = "reduce-overhead" if loss_cudagraphs else "default"
    if options.get("tensor_loss_fusion", False):
        structural_similarity = torch.compile(structural_similarity, backend="inductor", fullgraph=False, mode=compile_mode)
        # Normal backward replay changes the early refinement trajectory. Keep
        # its fusion, while restricting graph replay to the SSIM path.
        edge_aware_normal_loss = torch.compile(edge_aware_normal_loss, backend="inductor", fullgraph=False, mode="default")
        if loss_cudagraphs:
            ssim_forward_graph_replays = 0
            compiled_ssim = structural_similarity

            def observed_ssim(*args, **kwargs):
                nonlocal ssim_forward_graph_replays
                before = cudagraph_replays
                value = compiled_ssim(*args, **kwargs)
                ssim_forward_graph_replays += cudagraph_replays - before
                return value

            structural_similarity = observed_ssim
    plane_depth_calls = {"training": 0, "extraction": 0}
    plane_depth_phase = "training"
    plane_depth_first_call_seconds = 0.
    plane_depth_generated_kernels = 0
    if options.get("plane_depth_fusion", False):
        from torch._inductor import metrics as plane_metrics
        compiled_plane_depth = torch.compile(plane_depth, backend="inductor", fullgraph=True, mode="default")

        def observed_plane_depth(*args, **kwargs):
            nonlocal plane_depth_first_call_seconds, plane_depth_generated_kernels
            first = not any(plane_depth_calls.values())
            if first:
                tick = time.perf_counter()
                kernels_before = int(plane_metrics.generated_kernel_count)
            value = compiled_plane_depth(*args, **kwargs)
            if first:
                plane_depth_first_call_seconds = time.perf_counter() - tick
                plane_depth_generated_kernels = int(plane_metrics.generated_kernel_count) - kernels_before
            plane_depth_calls[plane_depth_phase] += 1
            return value

        plane_depth_function = observed_plane_depth
    preload_started = time.perf_counter()
    cache.preload(range(len(inputs.rows)))
    if options.get("ssim_target_cache", False):
        torch.cuda.synchronize()
    view_preload_seconds = time.perf_counter()-preload_started
    for step in range(updates, steps):
        if loss_cudagraphs:
            torch.compiler.cudagraph_mark_step_begin()
        if not stack:
            stack = inputs.train_indices.copy()
            random.shuffle(stack)
        index = stack.pop()
        item = cache.load(index)
        h, w = item["rgb"].shape[:2]
        layer = render_planar_gaussians(parameters, item["c2w"], item["K"], w, h, min(3, step // 1000),
                                       **camera_arguments(item))
        screen = layer["meta"].get("means2d")
        if step < 15000 and screen is not None and screen.requires_grad:
            screen.retain_grad()
        structural = 1 - structural_similarity(layer["rgb"], item["rgb"],
            item.get("ssim_target_mean"), item.get("ssim_target_variance"))
        ssim_forward_calls += 1
        adjusted = torch.where(structural.detach() < .5,
                               exposure[index, 0].exp() * layer["rgb"] + exposure[index, 1], layer["rgb"])
        loss = .8 * (adjusted - item["rgb"]).abs().mean() + .2 * structural
        loss = loss + pg_config.flatten_weight * layer["flatten_loss"]
        if step >= 1000:
            loss = loss + pg_config.normal_weight * edge_aware_normal_loss(
                layer["depth"], layer["valid"], layer["normal_sum_camera"],
                item["rgb"], item["c2w"], item["K"], item["foreground"],
                focal_lengths_checked=options.get("camera_geometry_cache", False),
                world_points=layer["points"] if options.get("reuse_render_points", False) else None,
                point_valid=layer["valid"] if options.get("reuse_render_points", False) else None)
        if step >= 1500 and step % 4 == 0 and graph[index]:
            multiview_updates += 1
            neighbor = cache.load(random.choice(graph[index]))
            with torch.no_grad():
                other = render_planar_gaussians(parameters, neighbor["c2w"], neighbor["K"],
                    neighbor["rgb"].shape[1], neighbor["rgb"].shape[0], min(3, step // 1000),
                    **camera_arguments(neighbor))
            consistency = multiview_consistency(item["rgb"], neighbor["rgb"],
                layer["points"], layer["normals"], layer["valid"],
                other["points"].detach(), other["normals"].detach(), other["valid"],
                item["foreground"], neighbor["foreground"], item["c2w"], neighbor["c2w"],
                item["K"], neighbor["K"], config=mv_config,
                reference_world_to_camera=item.get("world_to_camera"),
                neighbor_world_to_camera=neighbor.get("world_to_camera"),
                reference_intrinsics_inverse=item.get("intrinsics_inverse"),
                neighbor_intrinsics_inverse=neighbor.get("intrinsics_inverse"))
            loss = loss + weighted_multiview_regularization(consistency, mv_config)
            del other, neighbor, consistency
        loss.backward()
        validate_adam_inputs(loss, families, exposure)
        if step < 15000 and screen is not None and screen.grad is not None:
            visible = layer["meta"]["radii"][0]
            if visible.ndim == 2:
                visible = visible.max(-1).values
            visible = visible > 0
            gradient = screen.grad[0].detach().clone()
            gradient[..., 0] *= w / 2
            gradient[..., 1] *= h / 2
            sums += torch.linalg.vector_norm(gradient[..., :2], dim=-1) * visible
            counts += visible
        update_mean_learning_rate(optimizer, step)
        optimizer.step()
        updates += 1
        optimizer.zero_grad(set_to_none=True)
        if 500 <= step < 15000 and step % 100 == 0:
            refine_population(parameters, optimizer, "pgsr", sums, counts, max_points=options["max_points"])
            sums = torch.zeros(len(parameters["means"]), device="cuda")
            counts = torch.zeros_like(sums)
        if step > 0 and step % 3000 == 0:
            reset_opacities_if_due(parameters, optimizer, "pgsr", step, policy="refinement")
        if updates % 100 == 0 or updates == steps:
            if loss_cudagraphs and int(compiler_counters["inductor"].get("cudagraph_skips", 0)) > cudagraph_skips_before:
                raise RuntimeError("Requested loss CUDA graphs skipped execution")
            print(json.dumps(dict(stage="pgsr", updates=updates, steps=steps,
                                  loss=float(loss.detach()), points=len(parameters["means"]),
                                  cudagraph_replays=cudagraph_replays if loss_cudagraphs else None,
                                  elapsed_seconds=training_seconds_before + time.perf_counter() - started)), flush=True)
        del layer, screen, item, loss, structural, adjusted
        if time.perf_counter() - last_disk_check >= 30:
            try:
                check_disk()
            except RuntimeError:
                save()
                raise
            last_disk_check = time.perf_counter()
        if updates % 3000 == 0:
            check_disk()
            save()
    if loss_cudagraphs:
        CUDAGraphNode.run_graph = original_run_graph
        if updates > initial_updates and cudagraph_replays == 0:
            raise RuntimeError("Requested loss CUDA graphs did not replay")
    check_disk()
    save()
    torch.cuda.synchronize()
    training_seconds = training_seconds_before + time.perf_counter() - started
    if not export_on_finish:
        return None, dict(representation="pgsr_only", updates=updates, training_seconds=training_seconds)
    # Release cached full-resolution views before streaming TSDF integration.
    calibrations = ({index: {key: entry[key] for key in
                    ("c2w", "K", "world_to_camera", "intrinsics_inverse")}
                    for index, entry in cache.entries.items()}
                    if options.get("camera_geometry_cache", False) else {})
    target_statistics_views = sum("ssim_target_mean" in entry for entry in cache.entries.values())
    target_statistics_bytes = sum(entry[key].numel()*entry[key].element_size()
        for entry in cache.entries.values() for key in ("ssim_target_mean", "ssim_target_variance") if key in entry)
    cache.entries.clear()
    plane_depth_phase = "extraction"
    path, extraction_seconds = extract(parameters, updates, calibrations)
    up = -np.stack([r["c2w"][:3, 1] for r in inputs.rows]).mean(0)
    fusion_kernels = None
    if options.get("tensor_loss_fusion", False):
        from torch._inductor import metrics as compiler_metrics
        fusion_kernels = int(compiler_metrics.generated_kernel_count)
    return path, dict(representation="pgsr_only", updates=updates,
        points=len(parameters["means"]), training_seconds=training_seconds,
        multiview_updates=multiview_updates,
        extraction_seconds=extraction_seconds, camera_up=up.tolist(),
        normalized_colmap_radius=inputs.radius, normalized_colmap_center=inputs.center.tolist(),
        camera_quality=camera_quality, camera_precompute_views=len(calibrations),
        camera_precompute_seconds=camera_precompute_seconds,
        view_preload_seconds=view_preload_seconds, ssim_target_cache=options.get("ssim_target_cache", False),
        target_statistics_views=target_statistics_views, target_statistics_bytes=target_statistics_bytes,
        tensor_loss_fusion=options.get("tensor_loss_fusion", False), fusion_generated_kernels=fusion_kernels,
        plane_depth_fusion=options.get("plane_depth_fusion", False),
        plane_depth_calls=plane_depth_calls,
        plane_depth_first_call_seconds=plane_depth_first_call_seconds,
        plane_depth_generated_kernels=plane_depth_generated_kernels,
        loss_cudagraphs=loss_cudagraphs,
        separable_ssim=options.get("separable_ssim", False),
        ssim_forward_calls=ssim_forward_calls,
        separable_ssim_calls=ssim_forward_calls if options.get("separable_ssim", False) else 0,
        executed_training_updates=updates - initial_updates,
        cudagraph_replays=cudagraph_replays if loss_cudagraphs else 0,
        cudagraph_scope="ssim_only" if loss_cudagraphs else None,
        ssim_forward_graph_replays=ssim_forward_graph_replays if loss_cudagraphs else 0,
        cudagraph_skips=(int(compiler_counters["inductor"].get("cudagraph_skips", 0)) - cudagraph_skips_before
                        if loss_cudagraphs else None))


def main() -> None:
    import argparse
    import math

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-root", type=Path, help="Standalone PGSR source checkout")
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--factor", type=int, default=1, help="Image downscale factor")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normal-weight", type=float, default=0.015)
    parser.add_argument("--tsdf-voxel", type=float, default=0.002)
    parser.add_argument("--tsdf-trunc", type=float, default=0.008)
    parser.add_argument("--gpu-view-cache", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--studio", action="store_true")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--export-only", action="store_true")
    action.add_argument("--no-export", action="store_true")
    args = parser.parse_args()

    if args.steps < 1 or args.factor < 1 or not 0 <= args.seed < 2**32:
        parser.error("steps and factor must be positive, and seed must be a uint32")
    if not math.isfinite(args.normal_weight) or args.normal_weight < 0:
        parser.error("normal-weight must be finite and nonnegative")
    if not all(map(math.isfinite, (args.tsdf_voxel, args.tsdf_trunc))) or not (
        0 < args.tsdf_voxel <= args.tsdf_trunc
    ):
        parser.error("TSDF lengths must be finite and truncation must be at least one voxel")

    source = args.source_root
    if source is None and not os.environ.get("PGSR_SOURCE_ROOT"):
        source = Path(__file__).resolve().parent
        if not (source / "unbiased_gsplat").is_dir():
            parser.error("set --source-root or PGSR_SOURCE_ROOT to the standalone PGSR source")
    if not args.no_export:
        try:
            import open3d
        except ImportError:
            parser.error("install Open3D for PGSR mesh extraction")

    options = dict(
        steps=args.steps, factor=args.factor, seed=args.seed, max_points=0,
        normal_weight=args.normal_weight, tsdf_voxel=args.tsdf_voxel,
        tsdf_trunc=args.tsdf_trunc, gpu_view_cache=args.gpu_view_cache,
    )
    mesh, summary = train_and_extract(
        args.data.expanduser().resolve(), args.output, options, source=source,
        device=args.device, export_only=args.export_only, export_on_finish=not args.no_export,
        mesh_path=args.output / "mesh.ply" if args.studio else None,
    )
    print(json.dumps({"mesh": str(mesh) if mesh is not None else None, **summary}, indent=2))


if __name__ == "__main__":
    main()
