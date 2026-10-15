"""Readable tensor references derived from public mathematics, not target code.

All tensors stay on the supplied device/dtype. Deliberate stop-gradient rules
are part of the operator contract, not an implementation optimization.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def rotation_matrix(quats: torch.Tensor) -> torch.Tensor:
    w, x, y, z = F.normalize(quats, dim=-1).unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        -1,
    ).reshape(*quats.shape[:-1], 3, 3)


def covariance(quats: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    rotation = rotation_matrix(quats)
    return (rotation * scales.square().unsqueeze(-2)) @ rotation.transpose(-1, -2)


def ray_peak(means, inverse_covariances, origins, directions):
    delta = origins.unsqueeze(0) - means.unsqueeze(1)
    qd = torch.einsum("nij,pj->npi", inverse_covariances, directions)
    denominator = torch.einsum("pi,npi->np", directions, qd).clamp_min(1e-12)
    t = -(delta * qd).sum(-1) / denominator
    offset = delta + t.unsqueeze(-1) * directions.unsqueeze(0)
    exponent = torch.einsum("npi,nij,npj->np", offset, inverse_covariances, offset)
    response = torch.exp(-0.5 * exponent.clamp_min(0))
    return t, response, F.normalize(-qd, dim=-1)


def pixel_rays(K, *, width: int, height: int):
    y, x = torch.meshgrid(
        torch.arange(height, device=K.device, dtype=K.dtype) + 0.5,
        torch.arange(width, device=K.device, dtype=K.dtype) + 0.5,
        indexing="ij",
    )
    pixels = torch.stack((x, y, torch.ones_like(x)), -1)
    return pixels @ torch.linalg.inv(K).T


def depth_normals(depth, K):
    height, width = depth.shape
    points = pixel_rays(K, width=width, height=height) * depth.unsqueeze(-1)
    normals = torch.zeros_like(points)
    valid = torch.zeros_like(depth, dtype=torch.bool)
    if height < 3 or width < 3:
        return normals, valid
    dx = points[1:-1, 2:] - points[1:-1, :-2]
    dy = points[2:, 1:-1] - points[:-2, 1:-1]
    interior = F.normalize(torch.linalg.cross(dy, dx), dim=-1)
    facing = (interior * points[1:-1, 1:-1]).sum(-1, keepdim=True)
    interior = torch.where(facing > 0, -interior, interior)
    good = (
        (depth[1:-1, 1:-1] > 0)
        & (depth[1:-1, 2:] > 0)
        & (depth[1:-1, :-2] > 0)
        & (depth[2:, 1:-1] > 0)
        & (depth[:-2, 1:-1] > 0)
        & torch.isfinite(interior).all(-1)
        & (interior.square().sum(-1) > 0)
    )
    normals[1:-1, 1:-1] = torch.where(good.unsqueeze(-1), interior, 0)
    valid[1:-1, 1:-1] = good
    return normals, valid


def sh_color(coefficients, directions, *, degree: int):
    x, y, z = F.normalize(directions, dim=-1).unbind(-1)
    basis = [torch.full_like(x, 1 / math.sqrt(4 * math.pi))]
    if degree >= 1:
        c = math.sqrt(3 / (4 * math.pi))
        basis += [-c * y, c * z, -c * x]
    if degree >= 2:
        c = math.sqrt(15 / (4 * math.pi))
        basis += [
            c * x * y,
            -c * y * z,
            math.sqrt(5 / (16 * math.pi)) * (3 * z * z - 1),
            -c * x * z,
            math.sqrt(15 / (16 * math.pi)) * (x * x - y * y),
        ]
    if degree >= 3:
        basis += [
            -math.sqrt(35 / (32 * math.pi)) * y * (3 * x * x - y * y),
            math.sqrt(105 / (4 * math.pi)) * x * y * z,
            -math.sqrt(21 / (32 * math.pi)) * y * (5 * z * z - 1),
            math.sqrt(7 / (16 * math.pi)) * z * (5 * z * z - 3),
            -math.sqrt(21 / (32 * math.pi)) * x * (5 * z * z - 1),
            math.sqrt(105 / (16 * math.pi)) * z * (x * x - y * y),
            -math.sqrt(35 / (32 * math.pi)) * x * (x * x - 3 * y * y),
        ]
    return (
        0.5
        + (coefficients[:, : len(basis)] * torch.stack(basis, -1).unsqueeze(-1)).sum(1)
    ).clamp_min(0)


def sh_ambiguity(coefficients):
    return coefficients[:, 1:].square().sum((1, 2))


def ray_color_variance(weights, colors, composite):
    error = colors.unsqueeze(1) - composite.detach().unsqueeze(0)
    return (weights.detach() * error.square().sum(-1)).sum(0)


def alpha_composite(alphas, colors, depths, normals):
    alphas = alphas.clamp(0, 0.999)
    prefix = torch.cat((torch.ones_like(alphas[:1]), 1 - alphas), 0).cumprod(0)
    weights = alphas * prefix[:-1]
    alpha = weights.sum(0)
    rgb = torch.einsum("np,nc->pc", weights, colors)
    znum = (weights * depths).sum(0)
    nnum = (weights.unsqueeze(-1) * normals).sum(0)
    w = weights.detach()
    sorted_depth, order = depths.sort(0)
    sorted_w = weights.gather(0, order)
    pw = sorted_w.cumsum(0) - sorted_w
    pwd = (sorted_w * sorted_depth).cumsum(0) - sorted_w * sorted_depth
    l1 = (2 * sorted_w * (sorted_depth * pw - pwd)).sum(0).clamp_min(0)
    l2 = (
        2 * (w.sum(0) * (w * depths.square()).sum(0) - (w * depths).sum(0).square())
    ).clamp_min(0)
    if depths.shape[0]:
        crossed = weights.cumsum(0) >= 0.5
        index = crossed.to(torch.int64).argmax(0, keepdim=True)
        median = torch.where(crossed.any(0), depths.gather(0, index).squeeze(0), 0)
    else:
        median = torch.zeros_like(alpha)
    return rgb, alpha, znum, nnum, l1, l2, median, weights


def alpha_composite_batch(alphas, colors, depths, normals, *, distortions="both"):
    """Explicit B/N/P contract; no torch.func transform or hidden batch axis."""
    alphas = alphas.clamp(0, 0.999)
    prefix = torch.cat((torch.ones_like(alphas[:, :1]), 1 - alphas), 1).cumprod(1)
    weights = alphas * prefix[:, :-1]
    alpha = weights.sum(1)
    rgb = torch.einsum("bnp,bnc->bpc", weights, colors)
    znum = (weights * depths).sum(1)
    nnum = (weights.unsqueeze(-1) * normals).sum(1)
    w = weights.detach()
    l1 = torch.zeros_like(alpha)
    if distortions in ("l1", "both"):
        sorted_depth, order = depths.sort(1)
        sorted_w = weights.gather(1, order)
        pw = sorted_w.cumsum(1) - sorted_w
        pwd = (sorted_w * sorted_depth).cumsum(1) - sorted_w * sorted_depth
        l1 = (2 * sorted_w * (sorted_depth * pw - pwd)).sum(1).clamp_min(0)
    l2 = torch.zeros_like(alpha)
    if distortions in ("l2", "both"):
        l2 = (
            2 * (w.sum(1) * (w * depths.square()).sum(1) - (w * depths).sum(1).square())
        ).clamp_min(0)
    if depths.shape[1]:
        crossed = weights.cumsum(1) >= 0.5
        index = crossed.to(torch.int64).argmax(1, keepdim=True)
        median = torch.where(crossed.any(1), depths.gather(1, index).squeeze(1), 0)
    else:
        median = torch.zeros_like(alpha)
    return rgb, alpha, znum, nnum, l1, l2, median, weights


def alpha_composite_batch_selected(alphas, colors, depths, normals, *, distortions):
    """Version 3 has an explicit output-work selection, not a changed loss."""
    return alpha_composite_batch(
        alphas, colors, depths, normals, distortions=distortions
    )


def ray_color_variance_batch(weights, colors, composite):
    error = colors[:, :, None] - composite.detach()[:, None]
    return (weights.detach() * error.square().sum(-1)).sum(1)


def opacity_field(
    points,
    means,
    inverse_covariances,
    opacities,
    origins,
    *,
    gaussian_chunk: int = 1024,
):
    minimum = torch.ones(points.shape[0], device=points.device, dtype=points.dtype)
    for origin in origins:
        delta = points - origin
        distance = torch.linalg.vector_norm(delta, dim=-1)
        directions = F.normalize(delta, dim=-1)
        transmittance = torch.ones_like(minimum)
        for start in range(0, means.shape[0], gaussian_chunk):
            m = means[start : start + gaussian_chunk]
            q = inverse_covariances[start : start + gaussian_chunk]
            t, _, _ = ray_peak(m, q, origin.expand_as(points), directions)
            evaluated_t = torch.minimum(t, distance.unsqueeze(0))
            offset = origin - m.unsqueeze(1) + evaluated_t.unsqueeze(-1) * directions
            exponent = torch.einsum("npi,nij,npj->np", offset, q, offset)
            contribution = opacities[start : start + gaussian_chunk, None] * torch.exp(
                -0.5 * exponent.clamp_min(0)
            )
            contribution = torch.where(t > 0, contribution, 0).clamp(0, 0.999)
            transmittance = transmittance * (1 - contribution).prod(0)
        minimum = torch.minimum(minimum, 1 - transmittance)
    return minimum


def patch_zncc(a, b, mask, *, epsilon: float = 1e-8):
    count = mask.sum(-1).clamp_min(1)
    ac = a - (a * mask).sum(-1, keepdim=True) / count.unsqueeze(-1)
    bc = b - (b * mask).sum(-1, keepdim=True) / count.unsqueeze(-1)
    av = (mask * ac.square()).sum(-1)
    bv = (mask * bc.square()).sum(-1)
    valid = (mask.sum(-1) >= 2) & (av > epsilon) & (bv > epsilon)
    correlation = (mask * ac * bc).sum(-1) / (av * bv).clamp_min(
        epsilon * epsilon
    ).sqrt()
    return torch.where(valid, correlation.clamp(-1, 1), 0), valid
