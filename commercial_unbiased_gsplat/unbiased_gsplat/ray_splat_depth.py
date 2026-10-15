"""Differentiable gsplat-compatible ray/surfel intersection quantities."""

from __future__ import annotations

import torch
from torch import Tensor

from ._validation import (
    require_finite_scalar,
    require_floating,
    require_same_device,
    require_same_length,
)
from .models import RaySplatSamples


def ray_splat_depth_and_weight_torch(
    ray_transforms: Tensor,
    means2d: Tensor,
    pixel_ids: Tensor,
    image_width: int,
    *,
    hz_epsilon: float = 0.0,
) -> RaySplatSamples:
    """Recover per-hit Gaussian value and actual camera-space z-depth.

    The calculation follows the coordinate convention of gsplat commit
    ``6f378361``.  Invalid samples carry zero storage values and must be
    interpreted only through ``valid_mask``; zero is not a recovered depth.

    ``hz_epsilon=0`` mirrors gsplat's exact ``h.z == 0`` rejection.  A caller
    may opt into a larger numerical guard, but that changes the accepted
    boundary and must be recorded in experiment metadata.
    """

    if ray_transforms.ndim != 3 or ray_transforms.shape[-2:] != (3, 3):
        raise ValueError("ray_transforms must have shape [M, 3, 3]")
    if means2d.ndim != 2 or means2d.shape[-1] != 2:
        raise ValueError("means2d must have shape [M, 2]")
    if pixel_ids.ndim != 1:
        raise ValueError("pixel_ids must have shape [M]")
    if image_width <= 0:
        raise ValueError("image_width must be positive")
    hz_epsilon = require_finite_scalar("hz_epsilon", hz_epsilon)
    if hz_epsilon < 0.0:
        raise ValueError("hz_epsilon must be nonnegative")
    require_floating("ray_transforms", ray_transforms)
    require_floating("means2d", means2d)
    if means2d.dtype != ray_transforms.dtype:
        raise TypeError("means2d and ray_transforms must have the same dtype")
    require_same_length(
        (
            ("ray_transforms", ray_transforms),
            ("means2d", means2d),
            ("pixel_ids", pixel_ids),
        )
    )
    require_same_device(
        (
            ("ray_transforms", ray_transforms),
            ("means2d", means2d),
            ("pixel_ids", pixel_ids),
        )
    )

    scalar_dtype = ray_transforms.dtype
    pixel_x = torch.remainder(pixel_ids, image_width).to(scalar_dtype) + 0.5
    pixel_y = torch.div(pixel_ids, image_width, rounding_mode="floor").to(scalar_dtype) + 0.5
    pixel_coords = torch.stack((pixel_x, pixel_y), dim=-1)

    u_m = ray_transforms[:, 0, :]
    v_m = ray_transforms[:, 1, :]
    w_m = ray_transforms[:, 2, :]
    h_u = pixel_x[:, None] * w_m - u_m
    h_v = pixel_y[:, None] * w_m - v_m
    homogeneous = torch.cross(h_u, h_v, dim=-1)
    hz = homogeneous[:, 2]

    denominator_valid = torch.isfinite(homogeneous).all(dim=-1) & (hz.abs() > hz_epsilon)
    safe_hz = torch.where(denominator_valid, hz, torch.ones_like(hz))
    local_uv = homogeneous[:, :2] / safe_hz[:, None]
    local_uv = torch.where(denominator_valid[:, None], local_uv, torch.zeros_like(local_uv))

    rho3 = torch.sum(local_uv.square(), dim=-1)
    delta = means2d - pixel_coords
    rho2 = 2.0 * torch.sum(delta.square(), dim=-1)
    rho = torch.minimum(rho3, rho2)
    ray_plane_mask = denominator_valid & (rho3 <= rho2)
    ray_depth = local_uv[:, 0] * w_m[:, 0] + local_uv[:, 1] * w_m[:, 1] + w_m[:, 2]
    depth = torch.where(ray_plane_mask, ray_depth, w_m[:, 2])

    valid = (
        denominator_valid
        & torch.isfinite(rho3)
        & torch.isfinite(rho2)
        & torch.isfinite(rho)
        & torch.isfinite(depth)
        & (rho >= 0.0)
    )
    gaussian_weight = torch.exp(-0.5 * torch.where(valid, rho, torch.zeros_like(rho)))
    gaussian_weight = torch.where(valid, gaussian_weight, torch.zeros_like(gaussian_weight))
    depth = torch.where(valid, depth, torch.zeros_like(depth))
    rho3 = torch.where(valid, rho3, torch.zeros_like(rho3))
    rho2 = torch.where(valid, rho2, torch.zeros_like(rho2))
    local_uv = torch.where(valid[:, None], local_uv, torch.zeros_like(local_uv))
    ray_plane_mask = ray_plane_mask & valid

    return RaySplatSamples(
        gaussian_weight=gaussian_weight,
        depth=depth,
        rho3=rho3,
        rho2=rho2,
        local_uv=local_uv,
        ray_plane_mask=ray_plane_mask,
        valid_mask=valid,
    )


def ray_splat_depth_and_weight(
    ray_transforms: Tensor,
    means2d: Tensor,
    pixel_ids: Tensor,
    image_width: int,
    *,
    hz_epsilon: float = 0.0,
) -> RaySplatSamples:
    if ray_transforms.is_cuda:
        from ._cuda import cuda_available, ray_splat_depth_and_weight_fused

        if cuda_available():
            samples, _ = ray_splat_depth_and_weight_fused(
                ray_transforms,
                means2d,
                pixel_ids,
                image_width,
                backend="cuda",
                hz_epsilon=hz_epsilon,
            )
            return samples
    return ray_splat_depth_and_weight_torch(
        ray_transforms, means2d, pixel_ids, image_width, hz_epsilon=hz_epsilon
    )
