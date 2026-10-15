"""Stable regularized ray planes for the Apache gsplat RaDe-GS adapter.

Equations/regularization follow the pinned Apache-2.0 port's Utils.cuh;
the small SPD solve uses float64 Cholesky instead of float32 adjugate inversion.
Outputs and parameter gradients retain the caller's floating dtype.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .reference import covariance


def regularized_ray_geometry(
    means,
    quats,
    scales,
    world_to_camera,
    K,
    *,
    width,
    height,
    relative_jitter=1e-6,
    absolute_jitter=1e-8,
):
    dtype = means.dtype
    means, quats, scales, transform, K = (
        tensor.to(torch.float64)
        for tensor in (means, quats, scales, world_to_camera, K)
    )
    rotation = transform[:3, :3]
    local = means @ rotation.T + transform[:3, 3]
    x, y, z = local.unbind(-1)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = (x / z).clamp(
        -cx / fx - 0.15 * width / fx, (width - cx) / fx + 0.15 * width / fx
    )
    v = (y / z).clamp(
        -cy / fy - 0.15 * height / fy, (height - cy) / fy + 0.15 * height / fy
    )
    tx, ty = z * u, z * v
    distance = torch.sqrt(tx.square() + ty.square() + z.square())
    cov = rotation @ covariance(quats, scales) @ rotation.T
    trace = cov.diagonal(dim1=-2, dim2=-1).sum(-1)
    jitter = (relative_jitter * trace.clamp_min(absolute_jitter)).clamp_min(
        absolute_jitter
    )
    cov = (cov + cov.transpose(-1, -2)) * 0.5 + jitter[:, None, None] * torch.eye(
        3, device=cov.device, dtype=cov.dtype
    )
    ray = torch.stack((u, v, torch.ones_like(u)), -1)
    inverse_ray = torch.cholesky_solve(
        ray[..., None], torch.linalg.cholesky(cov)
    ).squeeze(-1)
    normalized = inverse_ray / (inverse_ray * ray).sum(-1, keepdim=True)
    plane = torch.stack(
        (
            (v.square() + 1) * normalized[:, 0]
            - u * v * normalized[:, 1]
            - u * normalized[:, 2],
            -u * v * normalized[:, 0]
            + (u.square() + 1) * normalized[:, 1]
            - v * normalized[:, 2],
        ),
        -1,
    )
    factor = distance / (u.square() + v.square() + 1)
    ray_normal = torch.cat((-plane * factor[:, None], -torch.ones_like(u[:, None])), -1)
    # Column-major GLM nJ_T in the pinned implementation, expressed row-major.
    camera_normal = torch.stack(
        (
            ray_normal[:, 0] / z + tx / distance * ray_normal[:, 2],
            ray_normal[:, 1] / z + ty / distance * ray_normal[:, 2],
            -tx / z.square() * ray_normal[:, 0]
            - ty / z.square() * ray_normal[:, 1]
            + z / distance * ray_normal[:, 2],
        ),
        -1,
    )
    planes = plane * factor[:, None] / torch.stack((fx, fy))
    return (
        torch.linalg.vector_norm(local, dim=-1).to(dtype),
        planes.to(dtype),
        F.normalize(camera_normal, dim=-1).to(dtype),
    )
