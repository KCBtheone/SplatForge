"""Optional fused CUDA increment behind an explicit backend flag.

Normative math stays in SPEC.md / the Torch implementation. This module
never silently substitutes: the used backend is always returned.
CUDA kernels are independently written under Apache-2.0.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor

from .models import RaySplatSamples
from .ray_splat_depth import ray_splat_depth_and_weight_torch

Backend = Literal["auto", "torch", "cuda"]

_BUILT = False
_BUILD_ERROR: str | None = None
AUX_DEPTH_VJP_CARRY = False
AUX_DEPTH_VJP_CARRY_CALLS = 0
AUX_FRONT_PREFIX_REPLAY = False
AUX_FRONT_PREFIX_REPLAY_CALLS = 0
AUX_FRONT_PREFIX_FORWARD = False
AUX_FRONT_PREFIX_FORWARD_CALLS = 0
AUXILIARY_STREAM_EXECUTION = None
SHARED_SURFEL_FORWARD_EXECUTION = None

try:
    import unbiased_gsplat_cuda  # type: ignore
except Exception as exc:
    unbiased_gsplat_cuda = None  # type: ignore
    _BUILT = False
    _BUILD_ERROR = f"{type(exc).__name__}: {exc}"
else:
    _BUILT = True
    _BUILD_ERROR = None


def cuda_available() -> bool:
    return bool(_BUILT) and torch.cuda.is_available()


def load_studio_cuda(source_dir: Path) -> None:
    """Reuse the installed extension or build the bundled kernels."""
    global unbiased_gsplat_cuda, _BUILT, _BUILD_ERROR
    if not torch.cuda.is_available():
        raise RuntimeError("Unbiased Surfel requires an NVIDIA CUDA GPU")
    required = ("rasterize_unbiased_aux", "rasterize_unbiased_aux_backward",
                "depth_to_points", "depth_to_points_backward")
    if _BUILT and all(hasattr(unbiased_gsplat_cuda, symbol) for symbol in required):
        return

    from torch.utils.cpp_extension import load
    from splatforge._paths import data_root, output_path

    name = "splatforge_unbiased_cuda"
    root = os.environ.get("TORCH_EXTENSIONS_DIR") or os.environ.get("SPLATFORGE_CUDA_BUILD_ROOT")
    build = None
    if root:
        build = output_path(Path(root) / name)
        build.mkdir(parents=True, exist_ok=True)
    elif data_root() is not None:
        raise RuntimeError("Set TORCH_EXTENSIONS_DIR to a directory inside SPLATFORGE_DATA_ROOT")
    sources = [source_dir / filename for filename in (
        "unbiased_fused_bind.cpp", "unbiased_fused.cu",
        "unbiased_aux.cu", "rasterize_unbiased_aux.cu",
    )]
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(f"Missing Unbiased Surfel CUDA source: {source}")
    unbiased_gsplat_cuda = load(
        name=name, sources=[str(source) for source in sources],
        extra_cflags=["/O2"] if os.name == "nt" else ["-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        build_directory=str(build) if build else None,
    )
    _BUILT = True
    _BUILD_ERROR = None


def get_build_info() -> dict:
    binary = Path(unbiased_gsplat_cuda.__file__).resolve() if _BUILT else None
    return {
        "built": bool(_BUILT),
        "cuda_available": cuda_available(),
        "build_error": _BUILD_ERROR,
        "torch_version": torch.__version__,
        "binary_path": str(binary) if binary else None,
        "binary_sha256": None,
        "binary_size_bytes": binary.stat().st_size if binary else None,
        "selective_depth_camera_backward": bool(_BUILT and getattr(unbiased_gsplat_cuda, "selective_depth_camera_backward", False)),
    }


class _RaySplatCuda(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        ray_transforms: Tensor,
        means2d: Tensor,
        pixel_ids: Tensor,
        image_width: int,
        hz_epsilon: float,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        ray_transforms = ray_transforms.contiguous()
        means2d = means2d.contiguous()
        pixel_ids = pixel_ids.contiguous()
        weight, depth, rho3, rho2, local_uv, plane, valid = unbiased_gsplat_cuda.ray_splat_forward(
            ray_transforms, means2d, pixel_ids, int(image_width), float(hz_epsilon)
        )
        ctx.image_width = int(image_width)
        ctx.save_for_backward(
            ray_transforms, means2d, pixel_ids, local_uv, weight, valid, plane
        )
        return weight, depth, rho3, rho2, local_uv, plane, valid

    @staticmethod
    def backward(ctx, g_weight, g_depth, g_rho3, g_rho2, g_uv, g_plane, g_valid):
        ray_transforms, means2d, pixel_ids, local_uv, weight, valid, plane = ctx.saved_tensors
        zeros = torch.zeros_like(weight)
        if g_weight is None:
            g_weight = zeros
        if g_depth is None:
            g_depth = zeros
        if g_rho3 is None:
            g_rho3 = zeros
        if g_rho2 is None:
            g_rho2 = zeros
        if g_uv is None:
            g_uv = torch.zeros_like(local_uv)
        g_ray, g_means = unbiased_gsplat_cuda.ray_splat_backward(
            ray_transforms,
            means2d,
            pixel_ids,
            local_uv,
            weight,
            valid,
            plane,
            g_weight.contiguous(),
            g_depth.contiguous(),
            g_rho3.contiguous(),
            g_rho2.contiguous(),
            g_uv.contiguous(),
            int(ctx.image_width),
        )
        return g_ray, g_means, None, None, None


def _pack(weight, depth, rho3, rho2, local_uv, plane, valid) -> RaySplatSamples:
    return RaySplatSamples(
        gaussian_weight=weight,
        depth=depth,
        rho3=rho3,
        rho2=rho2,
        local_uv=local_uv,
        ray_plane_mask=plane,
        valid_mask=valid,
    )


def ray_splat_depth_and_weight_fused(
    ray_transforms: Tensor,
    means2d: Tensor,
    pixel_ids: Tensor,
    image_width: int,
    *,
    backend: Backend = "auto",
    hz_epsilon: float = 0.0,
) -> tuple[RaySplatSamples, str]:
    if backend == "cuda" or (backend == "auto" and cuda_available() and ray_transforms.is_cuda):
        if not cuda_available():
            raise RuntimeError(
                "cuda backend requested but fused extension is not built: "
                f"{_BUILD_ERROR}"
            )
        if not ray_transforms.is_cuda:
            raise RuntimeError("cuda backend requires CUDA tensors")
        packed = _RaySplatCuda.apply(
            ray_transforms, means2d, pixel_ids, int(image_width), float(hz_epsilon)
        )
        return _pack(*packed), "cuda-fused"
    if backend in ("torch", "auto"):
        out = ray_splat_depth_and_weight_torch(
            ray_transforms, means2d, pixel_ids, image_width, hz_epsilon=hz_epsilon
        )
        return out, "torch-fallback"
    raise ValueError(f"unsupported backend: {backend!r}")


class _CorrectedDepthCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, depths, opacities, weights, segment_ids, valid, num_segments, eps, thresh):
        dense_depth, dense_valid, selected, contrib, cumulative = (
            unbiased_gsplat_cuda.corrected_depth_forward(
                depths.contiguous(),
                opacities.contiguous(),
                weights.contiguous(),
                segment_ids.contiguous(),
                valid.contiguous(),
                int(num_segments),
                float(eps),
                float(thresh),
            )
        )
        ctx.save_for_backward(selected, dense_valid)
        ctx.num_hits = int(depths.shape[0])
        return dense_depth, dense_valid, selected, contrib, cumulative

    @staticmethod
    def backward(ctx, g_depth, g_valid, g_selected, g_contrib, g_cum):
        selected, dense_valid = ctx.saved_tensors
        g_depths = g_depth.new_zeros((ctx.num_hits,))
        if g_depth is not None and dense_valid.any():
            g_depths[selected[dense_valid]] = g_depth[dense_valid]
        return g_depths, None, None, None, None, None, None, None


class _ConvergenceCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, depths, weights, segment_ids, valid, scene_radius, far_k):
        prev, curr, mask, per_pair = unbiased_gsplat_cuda.convergence_pairs(
            depths.contiguous(),
            weights.contiguous(),
            segment_ids.contiguous(),
            valid.contiguous(),
            float(scene_radius),
        )
        ctx.far_k = float(far_k)
        ctx.save_for_backward(depths, weights, prev, curr, mask)
        return per_pair, mask, prev, curr

    @staticmethod
    def backward(ctx, g_per_pair, g_mask, g_prev, g_curr):
        depths, weights, prev, curr, mask = ctx.saved_tensors
        g_depths = torch.zeros_like(depths)
        if g_per_pair is None or not mask.any():
            return g_depths, None, None, None, None, None
        p = prev[mask]
        c = curr[mask]
        gp = g_per_pair[mask]
        delta = depths[c] - depths[p]
        w = torch.minimum(weights[c], weights[p])
        common = 2.0 * delta * gp * w
        g_depths.index_add_(0, p, -common)
        g_depths.index_add_(0, c, common * ctx.far_k)
        return g_depths, None, None, None, None, None


def corrected_depth_cuda(depths, opacities, weights, segment_ids, valid, num_segments, opacity_epsilon, threshold):
    packed = _CorrectedDepthCuda.apply(
        depths, opacities, weights, segment_ids, valid, int(num_segments),
        float(opacity_epsilon), float(threshold),
    )
    return packed


def convergence_pairs_cuda(depths, weights, segment_ids, valid, scene_radius, far_k):
    return _ConvergenceCuda.apply(
        depths, weights, segment_ids, valid, float(scene_radius), float(far_k)
    )


class _ZnccCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, reference_patches, neighbor_patches, variance_epsilon):
        corr, valid = unbiased_gsplat_cuda.zncc_forward(
            reference_patches.contiguous(),
            neighbor_patches.contiguous(),
            float(variance_epsilon),
        )
        ctx.variance_epsilon = float(variance_epsilon)
        ctx.save_for_backward(reference_patches, neighbor_patches, valid)
        return corr, valid

    @staticmethod
    def backward(ctx, g_corr, g_valid):
        reference_patches, neighbor_patches, valid = ctx.saved_tensors
        if g_corr is None:
            return torch.zeros_like(reference_patches), torch.zeros_like(neighbor_patches), None
        g_a, g_b = unbiased_gsplat_cuda.zncc_backward(
            reference_patches.contiguous(),
            neighbor_patches.contiguous(),
            g_corr.contiguous(),
            valid.contiguous(),
            float(ctx.variance_epsilon),
        )
        return g_a, g_b, None


class _DepthToPointsCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, depth, valid, camera_to_world, intrinsics):
        points, point_valid = unbiased_gsplat_cuda.depth_to_points(
            depth.contiguous(),
            valid.contiguous(),
            camera_to_world.contiguous(),
            intrinsics.contiguous(),
        )
        ctx.save_for_backward(depth, point_valid, camera_to_world, intrinsics)
        return points, point_valid

    @staticmethod
    def backward(ctx, g_points, g_valid):
        depth, point_valid, camera_to_world, intrinsics = ctx.saved_tensors
        need_depth, _, need_camera, need_intrinsics = ctx.needs_input_grad
        if g_points is None:
            return (torch.zeros_like(depth) if need_depth else None, None,
                    torch.zeros_like(camera_to_world) if need_camera else None,
                    torch.zeros_like(intrinsics) if need_intrinsics else None)
        arguments = (
            depth.contiguous(),
            point_valid.contiguous(),
            camera_to_world.contiguous(),
            intrinsics.contiguous(),
            g_points.contiguous(),
        )
        if getattr(unbiased_gsplat_cuda, "selective_depth_camera_backward", False):
            arguments += (need_camera, need_intrinsics)
        g_depth, g_c2w, g_K = unbiased_gsplat_cuda.depth_to_points_backward(*arguments)
        return (g_depth if need_depth else None, None,
                g_c2w if need_camera else None, g_K if need_intrinsics else None)


def zncc_cuda(reference_patches, neighbor_patches, variance_epsilon):
    return _ZnccCuda.apply(reference_patches, neighbor_patches, float(variance_epsilon))


def depth_to_points_cuda(depth, valid, camera_to_world, intrinsics):
    return _DepthToPointsCuda.apply(depth, valid, camera_to_world, intrinsics)


class _FusedAuxRaster(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means2d,
        ray_transforms,
        opacities,
        tile_offsets,
        flatten_ids,
        image_width,
        image_height,
        tile_size,
        opacity_eps,
        threshold,
        scene_radius,
        far_k,
        opacity_independent=False,
        surface_midpoint=False,
        front_convergence=False,
        terminal_surface=False,
        coherent_midpoint=False,
    ):
        global AUX_FRONT_PREFIX_FORWARD_CALLS
        ctx.opacity_independent = bool(opacity_independent)
        ctx.surface_midpoint = bool(surface_midpoint)
        ctx.front_convergence = bool(front_convergence)
        ctx.terminal_surface = bool(terminal_surface)
        ctx.coherent_midpoint = bool(coherent_midpoint)
        if ctx.coherent_midpoint and (not ctx.surface_midpoint or not getattr(unbiased_gsplat_cuda, 'optional_coherent_midpoint', False)):
            raise RuntimeError('Coherent midpoint requires surface midpoint and its qualified native backend')
        if ctx.terminal_surface and not getattr(unbiased_gsplat_cuda, 'optional_terminal_surface', False):
            raise RuntimeError('Loaded native extension does not support terminal surface output')
        if ctx.front_convergence and not getattr(unbiased_gsplat_cuda, 'optional_front_convergence', False):
            raise RuntimeError('Loaded native extension does not support front-window convergence')
        if ctx.surface_midpoint and not getattr(unbiased_gsplat_cuda, 'optional_surface_midpoint', False):
            raise RuntimeError('Loaded native extension does not support midpoint surface output')
        if ctx.opacity_independent and not getattr(unbiased_gsplat_cuda, 'optional_opacity_independent_membership', False):
            raise RuntimeError('Loaded native extension does not support opacity-independent geometry membership')
        arguments = (
            means2d.contiguous().float(),
            ray_transforms.contiguous().float(),
            opacities.contiguous().float(),
            tile_offsets.contiguous(),
            flatten_ids.contiguous(),
            int(image_width),
            int(image_height),
            int(tile_size),
            float(opacity_eps),
            float(threshold),
            float(scene_radius),
        )
        if ctx.terminal_surface:
            arguments += (ctx.opacity_independent, ctx.surface_midpoint, ctx.front_convergence, True)
        elif ctx.front_convergence:
            arguments += (ctx.opacity_independent, ctx.surface_midpoint, True)
        elif ctx.surface_midpoint:
            arguments += (ctx.opacity_independent, True)
        elif ctx.opacity_independent:
            arguments += (True,)
        if SHARED_SURFEL_FORWARD_EXECUTION is not None:
            depth, valid, conv, sel = SHARED_SURFEL_FORWARD_EXECUTION.auxiliary(arguments)
        elif AUX_FRONT_PREFIX_FORWARD and ctx.front_convergence:
            import front_prefix_forward_cuda
            if not getattr(front_prefix_forward_cuda, 'exact_front_prefix_forward', False):
                raise RuntimeError('Explicit exact front-prefix forward backend is unavailable')
            depth, valid, conv, sel = front_prefix_forward_cuda.front_prefix_forward(*arguments)
            AUX_FRONT_PREFIX_FORWARD_CALLS += 1
        else:
            depth, valid, conv, sel = unbiased_gsplat_cuda.rasterize_unbiased_aux(*arguments)
        ctx.image_width = int(image_width)
        ctx.image_height = int(image_height)
        ctx.tile_size = int(tile_size)
        ctx.opacity_eps = float(opacity_eps)
        ctx.threshold = float(threshold)
        ctx.scene_radius = float(scene_radius)
        ctx.far_k = float(far_k)
        ctx.carry_depth_vjp = AUX_DEPTH_VJP_CARRY
        ctx.front_prefix_replay = AUX_FRONT_PREFIX_REPLAY and any(ctx.needs_input_grad[:3])
        if ctx.coherent_midpoint and ctx.front_prefix_replay:
            raise RuntimeError('Coherent midpoint must use the qualified generic auxiliary backward')
        if ctx.front_prefix_replay and (ctx.carry_depth_vjp or ctx.opacity_independent or
                not ctx.front_convergence or not ctx.surface_midpoint or not ctx.terminal_surface):
            raise RuntimeError('Front-prefix replay requires the current midpoint/front/terminal contract without carry or independent membership')
        if ctx.carry_depth_vjp and not getattr(unbiased_gsplat_cuda, 'optional_aux_depth_vjp_carry', False):
            raise RuntimeError('Requested auxiliary depth VJP carry is unavailable in the loaded native extension')
        ctx.save_for_backward(means2d, ray_transforms, opacities, tile_offsets, flatten_ids)
        return depth, valid, conv, sel

    @staticmethod
    def backward(ctx, g_depth, g_valid, g_conv, g_sel):
        global AUX_DEPTH_VJP_CARRY_CALLS, AUX_FRONT_PREFIX_REPLAY_CALLS
        means2d, ray_transforms, opacities, tile_offsets, flatten_ids = ctx.saved_tensors
        spatial = (int(tile_offsets.shape[0]), ctx.image_height, ctx.image_width)
        if g_depth is None:
            g_depth = torch.zeros(spatial, device=means2d.device, dtype=means2d.dtype)
        if g_conv is None:
            g_conv = torch.zeros(spatial, device=means2d.device, dtype=means2d.dtype)
        arguments = (
            means2d.contiguous().float(),
            ray_transforms.contiguous().float(),
            opacities.contiguous().float(),
            tile_offsets.contiguous(),
            flatten_ids.contiguous(),
            g_depth.contiguous().float(),
            g_conv.contiguous().float(),
            int(ctx.image_width),
            int(ctx.image_height),
            int(ctx.tile_size),
            float(ctx.opacity_eps),
            float(ctx.threshold),
            float(ctx.scene_radius),
            float(ctx.far_k),
        )
        if ctx.coherent_midpoint:
            arguments += (ctx.carry_depth_vjp, ctx.opacity_independent, ctx.front_convergence, ctx.terminal_surface, True)
        elif ctx.terminal_surface:
            arguments += (ctx.carry_depth_vjp, ctx.opacity_independent, ctx.front_convergence, True)
        elif ctx.front_convergence:
            arguments += (ctx.carry_depth_vjp, ctx.opacity_independent, True)
        elif ctx.opacity_independent:
            arguments += (ctx.carry_depth_vjp, True)
        elif ctx.carry_depth_vjp:
            arguments += (True,)
        if ctx.front_prefix_replay:
            import front_prefix_cuda
            if not front_prefix_cuda.exact_front_prefix_replay:
                raise RuntimeError('Front-prefix native contract is unavailable')
            g_means, g_ray, g_opac = front_prefix_cuda.front_prefix_backward(*arguments)
            AUX_FRONT_PREFIX_REPLAY_CALLS += 1
        else:
            g_means, g_ray, g_opac = unbiased_gsplat_cuda.rasterize_unbiased_aux_backward(*arguments)
        if ctx.carry_depth_vjp:
            AUX_DEPTH_VJP_CARRY_CALLS += 1
        gradients = (g_means, g_ray, g_opac, None, None, None, None, None, None, None, None, None)
        return gradients + (None,)*(len(ctx.needs_input_grad)-len(gradients))


class _RayDepthOrderedAuxRaster(torch.autograd.Function):
    @staticmethod
    def forward(ctx,means,rays,opac,offsets,ids,width,height,tile,eps,threshold,radius,far_k,surface_midpoint=False,terminal_surface=False):
        if not getattr(unbiased_gsplat_cuda,'optional_ray_depth_order',False):
            raise RuntimeError('Loaded native extension does not support physical ray-depth order')
        if int(offsets.shape[0]) != 1:
            raise ValueError('Physical ray-depth order currently requires one image')
        if surface_midpoint and not getattr(unbiased_gsplat_cuda,'optional_surface_midpoint',False):
            raise RuntimeError('Loaded native extension does not support midpoint surface output')
        if terminal_surface and not getattr(unbiased_gsplat_cuda,'optional_terminal_surface',False):
            raise RuntimeError('Loaded native extension does not support terminal surface output')
        ctx.terminal_surface = terminal_surface
        arguments=(
            means.contiguous().float(),rays.contiguous().float(),opac.contiguous().float(),
            offsets.contiguous(),ids.contiguous(),width,height,tile,eps,threshold,radius)
        if terminal_surface:arguments+=(surface_midpoint,True)
        elif surface_midpoint:arguments+=(True,)
        outputs=unbiased_gsplat_cuda.rasterize_unbiased_aux_ray_order(*arguments)
        ctx.save_for_backward(means,rays,opac,*outputs[4:])
        ctx.constants=(width,height,eps,threshold,radius,far_k)
        return tuple(outputs[:4])

    @staticmethod
    def backward(ctx,g_depth,g_valid,g_conv,g_selected):
        means,rays,opac,keys,values,prefix=ctx.saved_tensors
        width,height,eps,threshold,radius,far_k=ctx.constants
        if g_depth is None:g_depth=means.new_zeros((1,height,width))
        if g_conv is None:g_conv=means.new_zeros((1,height,width))
        arguments=(
            means.contiguous().float(),rays.contiguous().float(),opac.contiguous().float(),
            keys,values,prefix,g_depth.contiguous().float(),g_conv.contiguous().float(),
            width,height,eps,threshold,radius,far_k)
        if ctx.terminal_surface:arguments+=(True,)
        gradients=unbiased_gsplat_cuda.rasterize_unbiased_aux_ray_order_backward(*arguments)
        return tuple(gradients)+(None,)*(len(ctx.needs_input_grad)-len(gradients))


def rasterize_unbiased_aux(
    means2d,
    ray_transforms,
    opacities,
    tile_offsets,
    flatten_ids,
    image_width,
    image_height,
    tile_size,
    opacity_eps,
    threshold,
    scene_radius,
    far_k=1.25,
    *,
    opacity_independent=False,
    ray_depth_order=False,
    surface_midpoint=False,
    front_convergence=False,
    terminal_surface=False,
    coherent_midpoint=False,
):
    if any(type(value) is not bool for value in (opacity_independent,ray_depth_order,surface_midpoint,front_convergence,terminal_surface,coherent_midpoint)):
        raise TypeError('auxiliary options must be boolean')
    if coherent_midpoint and (not surface_midpoint or ray_depth_order):
        raise ValueError('Coherent midpoint is qualified for midpoint native-order training only')
    if ray_depth_order:
        if opacity_independent or AUX_DEPTH_VJP_CARRY or front_convergence:
            raise ValueError('Physical ray-depth order must be tested independently of other auxiliary options')
        return _RayDepthOrderedAuxRaster.apply(
            means2d,ray_transforms,opacities,tile_offsets,flatten_ids,int(image_width),int(image_height),int(tile_size),
            float(opacity_eps),float(threshold),float(scene_radius),float(far_k),surface_midpoint,terminal_surface)
    arguments = (
        means2d,
        ray_transforms,
        opacities,
        tile_offsets,
        flatten_ids,
        int(image_width),
        int(image_height),
        int(tile_size),
        float(opacity_eps),
        float(threshold),
        float(scene_radius),
        float(far_k),
        opacity_independent,
        surface_midpoint,
        front_convergence,
        terminal_surface,
        coherent_midpoint,
    )
    shared = SHARED_SURFEL_FORWARD_EXECUTION
    if shared is not None and getattr(shared, 'coupled_backward', False):
        if coherent_midpoint:raise ValueError('Coherent midpoint has no qualified coupled backward')
        if AUXILIARY_STREAM_EXECUTION is not None or AUX_DEPTH_VJP_CARRY:
            raise RuntimeError('Joint reverse auxiliary fields require their original serial consumer')
        return shared.coupled_auxiliary(arguments)
    if AUXILIARY_STREAM_EXECUTION is not None:
        return AUXILIARY_STREAM_EXECUTION.auxiliary(_FusedAuxRaster.apply, arguments)
    return _FusedAuxRaster.apply(*arguments)

    
class _PointNormalsCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, valid, origins, epsilon, orient):
        import point_normals_cuda as native
        if not getattr(native, 'owned_five_point_normal_stencil_vjp', False):
            raise RuntimeError('Explicit point-normal stencil backend is unavailable')
        points=points.contiguous()
        normals,mask,inverse=native.forward(points,valid.contiguous(),origins.contiguous(),float(epsilon),bool(orient))
        ctx.save_for_backward(points,normals,inverse)
        ctx.mark_non_differentiable(mask)
        return normals,mask

    @staticmethod
    def backward(ctx, gradient, mask_gradient):
        import point_normals_cuda as native
        points,normals,inverse=ctx.saved_tensors
        if gradient is None:return torch.zeros_like(points),None,None,None,None
        return native.backward(points,normals,inverse,gradient.contiguous()),None,None,None,None


def point_normals_cuda(points, valid, origins, epsilon, orient):
    return _PointNormalsCuda.apply(points,valid,origins,epsilon,orient)
