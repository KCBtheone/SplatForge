"""Optional fused CUDA increment behind an explicit backend flag.

Normative math stays in SPEC.md / the Torch implementation. This module
never silently substitutes: the used backend is always returned.
CUDA kernels are independently written under Apache-2.0.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from .models import RaySplatSamples
from .ray_splat_depth import ray_splat_depth_and_weight_torch

Backend = Literal["auto", "torch", "cuda"]

_BUILT = False
_BUILD_ERROR: str | None = None

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


def get_build_info() -> dict:
    return {
        "built": bool(_BUILT),
        "cuda_available": cuda_available(),
        "build_error": _BUILD_ERROR,
        "torch_version": torch.__version__,
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
        if g_points is None:
            return torch.zeros_like(depth), None, torch.zeros_like(camera_to_world), torch.zeros_like(intrinsics)
        g_depth, g_c2w, g_K = unbiased_gsplat_cuda.depth_to_points_backward(
            depth.contiguous(),
            point_valid.contiguous(),
            camera_to_world.contiguous(),
            intrinsics.contiguous(),
            g_points.contiguous(),
        )
        return g_depth, None, g_c2w, g_K


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
    ):
        depth, valid, conv, sel = unbiased_gsplat_cuda.rasterize_unbiased_aux(
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
        ctx.image_width = int(image_width)
        ctx.image_height = int(image_height)
        ctx.tile_size = int(tile_size)
        ctx.opacity_eps = float(opacity_eps)
        ctx.threshold = float(threshold)
        ctx.scene_radius = float(scene_radius)
        ctx.far_k = float(far_k)
        ctx.save_for_backward(means2d, ray_transforms, opacities, tile_offsets, flatten_ids)
        return depth, valid, conv, sel

    @staticmethod
    def backward(ctx, g_depth, g_valid, g_conv, g_sel):
        means2d, ray_transforms, opacities, tile_offsets, flatten_ids = ctx.saved_tensors
        spatial = (int(tile_offsets.shape[0]), ctx.image_height, ctx.image_width)
        if g_depth is None:
            g_depth = torch.zeros(spatial, device=means2d.device, dtype=means2d.dtype)
        if g_conv is None:
            g_conv = torch.zeros(spatial, device=means2d.device, dtype=means2d.dtype)
        g_means, g_ray, g_opac = unbiased_gsplat_cuda.rasterize_unbiased_aux_backward(
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
        return g_means, g_ray, g_opac, None, None, None, None, None, None, None, None, None


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
):
    return _FusedAuxRaster.apply(
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
    )
