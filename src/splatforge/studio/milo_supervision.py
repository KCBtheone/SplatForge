"""MILo Eq. (4), (6)--(9) and input-rendered depth fusion.

The primitives use the paper's sums. No implicit per-view/point normalization,
GT labels, or detached Gaussian/mesh supervision is introduced here.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .milo_raster import camera_vertices


@dataclass
class DepthObservation:
    camera: object
    depth: torch.Tensor
    valid: torch.Tensor


def _depth_image(value):
    return value.squeeze(-1) if value.ndim == 3 else value


def projected_depth(points, observation):
    """Sample a top-down rendered depth image at original pinhole coordinates."""
    camera = observation.camera
    view = camera_vertices(points, camera)
    z = view[:, 2]
    safe_z = torch.where(z > 0, z, torch.ones_like(z))
    projected = view @ camera.K.to(points).T
    u, v = (projected[:, :2] / safe_z[:, None]).unbind(-1)
    in_fov = (z > 0) & (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)
    grid = torch.stack((2 * u / camera.width - 1, 2 * v / camera.height - 1), -1)[None, None]
    image = _depth_image(observation.depth).to(points)
    valid_image = _depth_image(observation.valid).to(points)
    if image.shape != (camera.height, camera.width) or valid_image.shape != image.shape:
        raise ValueError("depth and validity must match the original camera resolution")
    # A site samples one pixel; no blending across foreground/background edges.
    depth = F.grid_sample(image[None, None], grid, mode="nearest", align_corners=False).reshape(-1)
    present = F.grid_sample(valid_image[None, None], grid, mode="nearest", align_corners=False).reshape(-1) > 0
    return z, depth, in_fov, present & torch.isfinite(depth) & (depth > 0)


@torch.no_grad()
def fuse_signed_distances(points, observations):
    """Mean(rendered Gaussian depth - site depth); unobserved sites stay marked."""
    total = points.new_zeros(len(points))
    count = torch.zeros(len(points), dtype=torch.long, device=points.device)
    for observation in observations:
        z, depth, in_fov, present = projected_depth(points, observation)
        valid = in_fov & present
        total += torch.where(valid, depth - z, 0)
        count += valid.to(torch.long)
    return total / count.clamp_min(1), count


@torch.no_grad()
def raw_sdf_initialization(signed_distances, distance_scale):
    """Explicit distance scale; nearest representable interior tanh endpoints."""
    scale = torch.as_tensor(distance_scale, dtype=signed_distances.dtype, device=signed_distances.device)
    if not bool(torch.isfinite(scale).all() & (scale > 0).all()):
        raise ValueError("SDF initialization needs a finite positive distance scale")
    limit = torch.nextafter(signed_distances.new_tensor(1.), signed_distances.new_tensor(0.))
    return torch.atanh((signed_distances / scale).clamp(-limit, limit))


@torch.no_grad()
def mesh_interior_labels(points, observations):
    """Inside only if occluded by a surface in every in-FOV training view.

    A background/hole ray cannot prove interior, and a point outside every view
    is unobserved rather than vacuously inside. These labels use mesh renders.
    """
    inside = torch.ones(len(points), dtype=torch.bool, device=points.device)
    seen = torch.zeros_like(inside)
    for observation in observations:
        z, depth, in_fov, present = projected_depth(points, observation)
        inside &= ~in_fov | (present & (z > depth))
        seen |= in_fov
    return inside & seen


def volumetric_normal_loss(rendered_normal, depth_normal, valid):
    if rendered_normal.shape != depth_normal.shape or rendered_normal.shape[-1] != 3 or valid.shape != rendered_normal.shape[:-1]:
        raise ValueError("normal maps need a matching spatial validity mask")
    a = torch.where(valid[..., None], rendered_normal, 0)
    b = torch.where(valid[..., None], depth_normal, 0)
    return torch.where(valid, 1 - (a * b).sum(-1), 0).sum()


def mesh_consistency_losses(gaussian_depth, depth_normal, mesh_render, valid):
    if valid.shape != mesh_render.valid.shape or _depth_image(gaussian_depth).shape != valid.shape:
        raise ValueError("Gaussian/mesh depth maps and validity must share a resolution")
    valid = valid & mesh_render.valid
    difference = torch.where(valid, _depth_image(gaussian_depth) - _depth_image(mesh_render.depth), 0)
    depth = torch.log1p(difference.abs()).sum()
    normal = volumetric_normal_loss(depth_normal, mesh_render.normals, valid)
    return {"mesh_depth": .05 * depth, "mesh_normal": .05 * normal}


def sdf_regularizers(raw_sdf, interior_labels):
    if raw_sdf.ndim != 2 or raw_sdf.shape[1] != 9 or interior_labels.shape != raw_sdf.shape:
        raise ValueError("nine raw SDF values and mesh-derived interior labels required per pivot")
    sdf = raw_sdf.tanh()
    erosion = sdf[:, 0].relu().sum()
    # BCE(sigmoid(-f),1) = softplus(f); outside sites have no BCE term.
    interior = torch.where(interior_labels, F.softplus(sdf), 0).sum()
    return {"mesh_erosion": .005 * erosion, "mesh_interior": .005 * interior}
