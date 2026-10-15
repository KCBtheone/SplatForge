"""Independent TSDF and GOF exporters, with explicit protocol metadata."""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import torch

from ..mesh.io import write_mesh
from ..mesh.tetrahedra import marching_tetrahedra, triangulate
from ..providers.torch.gaussian.reference import rotation_matrix
from .artifacts import StorageManifest, atomic_json, timestamp
from .multiview import sample_image
from .plugins import ComponentRegistry
from .resources import wait_for_capacity


def gaussian_grid(model, config):
    opacity = model.opacities.detach().cpu().numpy()
    chosen = np.flatnonzero(opacity >= config.seed_opacity_min)
    if not len(chosen):
        raise ValueError("no Gaussian passes the mesh seed opacity threshold")
    if config.max_gaussians is not None and len(chosen) > config.max_gaussians:
        # Reduction is an explicit exporter setting, never an unrecorded OOM fallback.
        chosen = chosen[np.argsort(opacity[chosen])[-config.max_gaussians :]]
    indices = torch.as_tensor(chosen, device=model.device)
    means = model.means[indices].detach().cpu().numpy()
    scales = model.scales[indices].detach().cpu().numpy()
    rotation = rotation_matrix(model.quats[indices]).detach().cpu().numpy()
    signs = np.asarray([(0.0, 0.0, 0.0), *list(product((-1.0, 1.0), repeat=3))])
    local = signs[None] * scales[:, None] * config.tetra_extent
    points = means[:, None] + np.einsum("nij,nkj->nki", rotation, local)
    flat = points.reshape(-1, 3)
    unique, inverse = np.unique(flat, axis=0, return_inverse=True)
    radii = np.zeros(len(unique))
    np.maximum.at(radii, inverse, np.repeat(scales.max(-1) * config.tetra_extent, 9))
    return unique, triangulate(unique, radii), len(chosen)


class GOFExporter:
    @torch.no_grad()
    def export(self, pipeline, path, config):
        points, cells, seeds = gaussian_grid(pipeline.model, config)
        inverse = torch.linalg.inv(
            pipeline.ops(
                "gaussian.covariance@1", pipeline.model.quats, pipeline.model.scales
            )
        )
        origins = torch.stack(
            [
                pipeline.data.frames[i].camera.origin.to(pipeline.model.device)
                for i in pipeline.data.train_indices
            ]
        )

        def field(coordinates):
            rows = []
            for start in range(0, len(coordinates), config.query_chunk):
                sample = torch.as_tensor(
                    coordinates[start : start + config.query_chunk],
                    device=pipeline.model.device,
                    dtype=pipeline.model.means.dtype,
                )
                values = pipeline.ops(
                    "gaussian.opacity_field@1",
                    sample,
                    pipeline.model.means,
                    inverse,
                    pipeline.model.opacities,
                    origins,
                    gaussian_chunk=config.gaussian_chunk,
                )
                rows.append(values.cpu().numpy())
            return np.concatenate(rows) if rows else np.empty(0)

        values = field(points)
        vertices, faces = marching_tetrahedra(
            points,
            cells,
            values,
            field,
            level=config.level,
            binary_steps=config.binary_steps,
            inside_greater=True,
        )
        if not len(faces):
            raise ValueError("GOF level set is empty; training output is preserved")
        write_mesh(path, vertices, faces)
        return {
            "exporter": "gof",
            "vertices": len(vertices),
            "faces": len(faces),
            "grid_vertices": len(points),
            "tetrahedra": len(cells),
            "seed_gaussians": seeds,
            "level": config.level,
            "binary_steps": config.binary_steps,
            "training_views": len(origins),
            "path": str(path),
        }


class TSDFExporter:
    @torch.no_grad()
    def export(self, pipeline, path, config):
        if config.tsdf_backend == "open3d":
            return self._open3d(pipeline, path, config)
        if config.tsdf_backend != "adaptive":
            raise ValueError("TSDF backend must be adaptive or open3d")
        points, cells, seeds = gaussian_grid(pipeline.model, config)
        views = []
        for index in pipeline.data.train_indices:
            sample = pipeline.data.sample(index, pipeline.model.device)
            render = pipeline.render(sample["camera"])
            views.append(
                (sample["camera"], render.depth.detach(), render.alpha.detach())
            )

        def field(coordinates):
            rows = []
            for start in range(0, len(coordinates), config.query_chunk):
                world = torch.as_tensor(
                    coordinates[start : start + config.query_chunk],
                    device=pipeline.model.device,
                    dtype=pipeline.model.means.dtype,
                )
                accumulator = torch.zeros(len(world), device=world.device)
                weight = torch.zeros_like(accumulator)
                for camera, depth, alpha in views:
                    local = (
                        world @ camera.world_to_camera[:3, :3].T
                        + camera.world_to_camera[:3, 3]
                    )
                    projected = local @ camera.K.T
                    pixels = projected[:, :2] / projected[:, 2:3].clamp_min(1e-8)
                    observed = sample_image(depth, pixels).squeeze(-1)
                    confidence = sample_image(alpha, pixels).squeeze(-1)
                    delta = observed - local[:, 2]
                    valid = (
                        (local[:, 2] > 0)
                        & (local[:, 2] < config.depth_max)
                        & (observed > 0)
                        & (pixels[:, 0] >= 0.5)
                        & (pixels[:, 0] < camera.width - 0.5)
                        & (pixels[:, 1] >= 0.5)
                        & (pixels[:, 1] < camera.height - 0.5)
                        & (delta >= -config.truncation)
                        & (confidence >= config.alpha_min)
                    )
                    w = valid.to(confidence.dtype) * confidence
                    accumulator += (delta / config.truncation).clamp(-1, 1) * w
                    weight += w
                result = torch.where(
                    weight > 0,
                    accumulator / weight.clamp_min(1e-8),
                    torch.ones_like(weight),
                )
                rows.append(result.cpu().numpy())
            return np.concatenate(rows) if rows else np.empty(0)

        values = field(points)
        vertices, faces = marching_tetrahedra(
            points,
            cells,
            values,
            field,
            level=0.0,
            binary_steps=config.binary_steps,
            inside_greater=False,
        )
        if not len(faces):
            raise ValueError("TSDF surface is empty; training output is preserved")
        write_mesh(path, vertices, faces)
        return {
            "exporter": "tsdf-adaptive",
            "vertices": len(vertices),
            "faces": len(faces),
            "grid_vertices": len(points),
            "tetrahedra": len(cells),
            "seed_gaussians": seeds,
            "truncation": config.truncation,
            "mesher": "adaptive tetrahedra; not uniform Open3D voxel fusion",
            "path": str(path),
        }

    def _open3d(self, pipeline, path, config):
        try:
            import open3d as o3d
        except ImportError as exc:
            raise RuntimeError(
                "install the mesh extra for Open3D TSDF; no exporter fallback is performed"
            ) from exc
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=config.voxel_size,
            sdf_trunc=config.truncation,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )
        progress_path = Path(path).parent / "export_progress.json"
        for completed, index in enumerate(pipeline.data.train_indices, 1):
            sample = pipeline.data.sample(index, pipeline.model.device)
            camera = sample["camera"]
            render = pipeline.render(camera)
            color = (render.rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            depth = render.depth.squeeze(-1).cpu().numpy().astype(np.float32)
            depth[render.alpha.squeeze(-1).cpu().numpy() < config.alpha_min] = 0
            depth[~np.isfinite(depth) | (depth <= 0)] = 0
            if config.mask_background:
                depth[sample["mask"].squeeze(-1).cpu().numpy() <= 0] = 0
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(color),
                o3d.geometry.Image(depth),
                depth_scale=1.0,
                depth_trunc=config.depth_max,
                convert_rgb_to_intensity=False,
            )
            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                camera.width,
                camera.height,
                float(camera.K[0, 0]),
                float(camera.K[1, 1]),
                float(camera.K[0, 2]),
                float(camera.K[1, 2]),
            )
            volume.integrate(rgbd, intrinsic, camera.world_to_camera.cpu().numpy())
            atomic_json(
                progress_path,
                {
                    "stage": "FUSING",
                    "updated": timestamp(),
                    "completed_training_views": completed,
                    "total_training_views": len(pipeline.data.train_indices),
                    "last_view": camera.name,
                    "voxel_size": config.voxel_size,
                    "mask_background": config.mask_background,
                },
            )
        atomic_json(
            progress_path,
            {
                "stage": "MARCHING_CUBES",
                "updated": timestamp(),
                "completed_training_views": len(pipeline.data.train_indices),
            },
        )
        mesh = volume.extract_triangle_mesh()
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        if not len(faces):
            raise ValueError("Open3D TSDF surface is empty")
        write_mesh(path, vertices, faces)
        return {
            "exporter": "tsdf-open3d",
            "vertices": len(vertices),
            "faces": len(faces),
            "voxel_size": config.voxel_size,
            "truncation": config.truncation,
            "mask_background": config.mask_background,
            "path": str(path),
        }


exporters = ComponentRegistry("splatforge.exporters")
exporters.register("gof", GOFExporter)
exporters.register("tsdf", TSDFExporter)


@torch.no_grad()
def export_pipeline(pipeline, output, config=None):
    config = config or pipeline.config.export
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = StorageManifest(output)

    def stage(name, **extra):
        atomic_json(
            output / "export_state.json",
            {"stage": name, "updated": timestamp(), **extra},
        )

    wait_for_capacity(pipeline.config, stage, check_gpu=False)
    representation = pipeline.save_representation(output)
    manifest.record(
        representation["path"], "FINAL", "trained scene representation", state="READY"
    )
    kind = pipeline.method.exporter if config.kind == "method" else config.kind
    if kind == "gaussians":
        report = {"exporter": "gaussians", **representation}
    else:
        mesh_path = output / "mesh.ply"
        manifest.record(mesh_path, "FINAL", "reconstructed final mesh")
        stage("EXTRACTING", exporter=kind)
        try:
            report = exporters.get(kind)().export(pipeline, mesh_path, config)
        except BaseException as exc:
            stage("FAILED", exporter=kind, error=f"{type(exc).__name__}: {exc}")
            raise
        manifest.record(mesh_path, "FINAL", "reconstructed final mesh", state="READY")
    atomic_json(output / "export.json", report)
    stage("COMPLETE", **report)
    return report
