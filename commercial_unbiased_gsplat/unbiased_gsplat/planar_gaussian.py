"""Paper-derived PGSR planes on Apache gsplat's 3D rasterizer.

SPDX-License-Identifier: Apache-2.0
Equations 1--6 and 13--16 of arXiv:2406.06521v2 define this module.
No PGSR software, binary, checkpoint, or renderer is a dependency.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import Tensor
from .depth_normal import depth_to_normal, depth_to_points


@dataclass(frozen=True)
class PlanarGaussianConfig:
    flatten_weight: float = 100.0
    normal_weight: float = .015
    alpha_threshold: float = .01
    denominator_epsilon: float = 1e-8


def quaternion_rotation(quaternions: Tensor) -> Tensor:
    """Unit wxyz quaternion to a right-handed rotation, with autograd."""
    q = F.normalize(quaternions, dim=-1, eps=1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack((
        1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y),
    ), -1).reshape(q.shape[:-1] + (3, 3))


def primitive_plane_features(means: Tensor, quaternions: Tensor,
                             scales: Tensor, world_to_camera: Tensor) -> tuple[Tensor, Tensor]:
    if scales.shape != means.shape or means.shape[-1] != 3:
        raise ValueError('PGSR requires three physical scales per primitive')
    rotations = quaternion_rotation(quaternions)
    minimum_axes = scales.argmin(-1)
    world_normals = rotations.gather(-1, minimum_axes[:, None, None].expand(-1, 3, 1))[..., 0]
    camera_means = means @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    camera_normals = world_normals @ world_to_camera[:3, :3].T
    facing = (camera_means * camera_normals).sum(-1) > 0
    camera_normals = torch.where(facing[:, None], -camera_normals, camera_normals)
    signed_distances = (camera_means * camera_normals).sum(-1, keepdim=True)
    return camera_normals, signed_distances


def plane_depth(normal_sum: Tensor, signed_distance_sum: Tensor,
                alpha: Tensor, intrinsics: Tensor, *,
                config: PlanarGaussianConfig = PlanarGaussianConfig()) -> tuple[Tensor, Tensor]:
    """Ratio of identically alpha-blended plane distance and ray-dot-normal.

    Signed numerator and denominator share orientation. Alpha cancels; this
    is neither a Gaussian-center expected depth nor a median depth.
    """
    h, w = normal_sum.shape[:2]
    yy, xx = torch.meshgrid(torch.arange(h, device=normal_sum.device, dtype=normal_sum.dtype)+.5,
                            torch.arange(w, device=normal_sum.device, dtype=normal_sum.dtype)+.5,
                            indexing='ij')
    rays = torch.stack((xx, yy, torch.ones_like(xx)), -1) @ torch.linalg.inv(intrinsics).T
    denominator = (normal_sum * rays).sum(-1)
    finite = torch.isfinite(normal_sum).all(-1) & torch.isfinite(signed_distance_sum)
    eligible = finite & (denominator.abs() > config.denominator_epsilon) & (alpha > config.alpha_threshold)
    safe_denominator = torch.where(eligible, denominator, torch.ones_like(denominator))
    ratio = torch.where(finite, signed_distance_sum, torch.zeros_like(signed_distance_sum)) / safe_denominator
    valid = eligible & torch.isfinite(ratio) & (ratio > 0)
    return torch.where(valid, ratio, torch.zeros_like(ratio)), valid


def edge_aware_normal_loss(depth: Tensor, valid: Tensor, normal_sum_camera: Tensor,
                           image: Tensor, camera_to_world: Tensor, intrinsics: Tensor,
                           support: Tensor | None = None) -> Tensor:
    geometric = depth_to_normal(depth[None], valid[None], camera_to_world[None], intrinsics[None])
    accumulated_world = normal_sum_camera @ camera_to_world[:3, :3].T
    rendered = F.normalize(accumulated_world, dim=-1, eps=1e-12)
    # Normalize observed RGB gradient globally to [0, 1], as in public Eq.6.
    gradient = torch.zeros_like(depth)
    dx = (image[:, 1:] - image[:, :-1]).abs().mean(-1)
    dy = (image[1:] - image[:-1]).abs().mean(-1)
    gradient[:, 1:] = torch.maximum(gradient[:, 1:], dx)
    gradient[1:] = torch.maximum(gradient[1:], dy)
    gradient = gradient / gradient.max().clamp_min(1e-8)
    mask = geometric.valid_mask[0]
    if support is not None:
        mask = mask & support
    differences = (geometric.normals[0] - rendered).abs().sum(-1)
    weights = (1-gradient.detach()).square() * mask
    return (differences * weights).sum() / mask.sum().clamp_min(1)


def render_planar_gaussians(parameters: dict[str, Tensor], camera_to_world: Tensor,
                            intrinsics: Tensor, width: int, height: int,
                            sh_degree: int, *, config: PlanarGaussianConfig = PlanarGaussianConfig()) -> dict:
    from gsplat import rasterization
    from gsplat.cuda._wrapper import spherical_harmonics
    means, quats = parameters['means'], parameters['quats']
    scales, opacities = parameters['scales'].exp(), parameters['opacities'].sigmoid()
    view = torch.linalg.inv(camera_to_world)
    normals, distances = primitive_plane_features(means, quats, scales, view)
    sh = torch.cat((parameters['sh0'], parameters['shN']), 1)
    directions = means - camera_to_world[:3, 3]
    rgb = (spherical_harmonics(sh_degree, directions, sh) + .5).clamp_min(0)
    attributes = torch.cat((rgb, normals, distances), -1)
    values, alphas, meta = rasterization(means, quats, scales, opacities, attributes,
                                        view[None], intrinsics[None], width, height,
                                        packed=False, render_mode='RGB', near_plane=.01)
    values, alpha = values[0], alphas[0, ..., 0]
    depth, valid = plane_depth(values[..., 3:6], values[..., 6], alpha, intrinsics, config=config)
    points, point_valid = depth_to_points(depth[None], valid[None], camera_to_world[None], intrinsics[None])
    world_normals = F.normalize(values[..., 3:6] @ camera_to_world[:3, :3].T, dim=-1, eps=1e-12)
    return dict(rgb=values[..., :3], alpha=alpha, depth=depth, valid=valid & point_valid[0],
                points=points[0], normals=world_normals, normal_sum_camera=values[..., 3:6],
                distance_sum=values[..., 6], meta=meta,
                flatten_loss=scales.min(-1).values.mean())
