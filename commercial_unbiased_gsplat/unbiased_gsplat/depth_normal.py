"""Backprojection and validity-aware finite-difference depth normals."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ._validation import require_finite_scalar
from .models import DepthNormalResult


def depth_to_points(
    depth: Tensor,
    valid_mask: Tensor,
    camera_to_world: Tensor,
    intrinsics: Tensor,
) -> tuple[Tensor, Tensor]:
    """Backproject camera z-depth at pixel centers into world coordinates."""

    if depth.ndim < 2:
        raise ValueError("depth must have shape [..., H, W] or [..., H, W, 1]")
    if depth.shape[-1:] == (1,) and depth.ndim >= 3:
        depth = depth[..., 0]
    if valid_mask.shape[-1:] == (1,) and valid_mask.ndim == depth.ndim + 1:
        valid_mask = valid_mask[..., 0]
    if depth.shape != valid_mask.shape:
        raise ValueError("depth and valid_mask shapes must match")
    if valid_mask.dtype != torch.bool:
        raise TypeError("valid_mask must be boolean")
    if not depth.dtype.is_floating_point:
        raise TypeError("depth must have a floating dtype")
    if camera_to_world.shape[-2:] != (4, 4) or intrinsics.shape[-2:] != (3, 3):
        raise ValueError("camera_to_world/intrinsics must end in [4,4]/[3,3]")
    if depth.shape[:-2] != camera_to_world.shape[:-2] or depth.shape[:-2] != intrinsics.shape[:-2]:
        raise ValueError("camera batch dimensions must match depth")
    if depth.device != camera_to_world.device or depth.device != intrinsics.device:
        raise ValueError("all inputs must share a device")
    if depth.dtype != camera_to_world.dtype or depth.dtype != intrinsics.dtype:
        raise TypeError("all floating inputs must share dtype")

    if depth.is_cuda:
        from ._cuda import cuda_available, depth_to_points_cuda

        if cuda_available():
            return depth_to_points_cuda(depth, valid_mask, camera_to_world, intrinsics)

    height, width = int(depth.shape[-2]), int(depth.shape[-1])
    y, x = torch.meshgrid(
        torch.arange(height, dtype=depth.dtype, device=depth.device),
        torch.arange(width, dtype=depth.dtype, device=depth.device),
        indexing="ij",
    )
    fx = intrinsics[..., 0, 0]
    fy = intrinsics[..., 1, 1]
    cx = intrinsics[..., 0, 2]
    cy = intrinsics[..., 1, 2]
    if bool(torch.any(fx == 0).item()) or bool(torch.any(fy == 0).item()):
        raise ValueError("camera focal lengths must be nonzero")
    camera_dirs = torch.stack(
        (
            (x - cx[..., None, None] + 0.5) / fx[..., None, None],
            (y - cy[..., None, None] + 0.5) / fy[..., None, None],
            torch.ones_like(depth),
        ),
        dim=-1,
    )
    camera_points = depth[..., None] * camera_dirs
    rotation = camera_to_world[..., :3, :3]
    origin = camera_to_world[..., :3, 3]
    world_points = torch.einsum("...ij,...hwj->...hwi", rotation, camera_points)
    world_points = world_points + origin[..., None, None, :]
    finite = torch.isfinite(world_points).all(dim=-1) & torch.isfinite(depth)
    point_valid = valid_mask & finite
    world_points = torch.where(
        point_valid[..., None], world_points, torch.zeros_like(world_points)
    )
    return world_points, point_valid


def depth_to_normal(
    depth: Tensor,
    valid_mask: Tensor,
    camera_to_world: Tensor,
    intrinsics: Tensor,
    *,
    normal_epsilon: float = 1e-12,
    orient_toward_camera: bool = True,
) -> DepthNormalResult:
    """Compute central-difference normals with a five-pixel validity stencil."""

    normal_epsilon = require_finite_scalar("normal_epsilon", normal_epsilon)
    if normal_epsilon < 0.0:
        raise ValueError("normal_epsilon must be nonnegative")
    points, point_valid = depth_to_points(depth, valid_mask, camera_to_world, intrinsics)
    height, width = int(points.shape[-3]), int(points.shape[-2])
    normals = torch.zeros_like(points)
    normal_valid = torch.zeros_like(point_valid)
    if height < 3 or width < 3:
        return DepthNormalResult(points=points, normals=normals, valid_mask=normal_valid)

    dpoint_dx = points[..., 1:-1, 2:, :] - points[..., 1:-1, :-2, :]
    dpoint_dy = points[..., 2:, 1:-1, :] - points[..., :-2, 1:-1, :]
    cross = torch.cross(dpoint_dx, dpoint_dy, dim=-1)
    magnitude = torch.linalg.vector_norm(cross, dim=-1)
    stencil_valid = (
        point_valid[..., 1:-1, 1:-1]
        & point_valid[..., 1:-1, :-2]
        & point_valid[..., 1:-1, 2:]
        & point_valid[..., :-2, 1:-1]
        & point_valid[..., 2:, 1:-1]
    )
    interior_valid = (
        stencil_valid
        & torch.isfinite(cross).all(dim=-1)
        & torch.isfinite(magnitude)
        & (magnitude > normal_epsilon)
    )
    safe_magnitude = torch.where(interior_valid, magnitude, torch.ones_like(magnitude))
    interior_normals = cross / safe_magnitude[..., None]

    if orient_toward_camera:
        camera_origins = camera_to_world[..., :3, 3][..., None, None, :]
        to_camera = camera_origins - points[..., 1:-1, 1:-1, :]
        facing = torch.sum(interior_normals * to_camera, dim=-1)
        interior_normals = torch.where(
            (facing < 0.0)[..., None], -interior_normals, interior_normals
        )
    interior_normals = torch.where(
        interior_valid[..., None], interior_normals, torch.zeros_like(interior_normals)
    )
    normals = F.pad(interior_normals, (0, 0, 1, 1, 1, 1), value=0.0)
    normal_valid = F.pad(interior_valid, (1, 1, 1, 1), value=False)
    return DepthNormalResult(points=points, normals=normals, valid_mask=normal_valid)
