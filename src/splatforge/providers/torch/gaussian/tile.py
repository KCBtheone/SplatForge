"""Pure batched tile equations for an explicitly selected compiling backend."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .reference import alpha_composite_batch, ray_color_variance_batch


def checkpointed_tile_batch(*args, **kwargs):
    """Compile the checkpoint together with its pure, uncompiled inner math.

    In Torch 2.1, wrapping an already Dynamo-optimized callable in checkpoint
    makes AOTAutograd recursively FX-trace Dynamo during backward. The inverse
    composition lets Dynamo represent recomputation as a higher-order operator.
    """
    return checkpoint(
        render_tile_batch,
        *args,
        use_reentrant=False,
        preserve_rng_state=False,
        **kwargs,
    )


def render_tile_batch(
    delta,
    pixel_means,
    q2,
    color,
    ray_color,
    opacity,
    means,
    normal,
    gain,
    q3,
    plane_distance,
    valid_slots,
    selection,
    rays,
    background,
    *,
    mode,
    near,
    far,
    extent,
    distortions="both",
):
    squared = torch.einsum("blpi,blij,blpj->blp", delta, q2, delta)
    if mode == "gof":
        qd = torch.einsum("blij,bpj->blpi", q3, rays)
        denominator = (qd * rays[:, None]).sum(-1).clamp_min(1e-12)
        depth = (pixel_means * qd).sum(-1) / denominator
        offset = depth[..., None] * rays[:, None] - pixel_means
        exponent = torch.einsum("blpi,blij,blpj->blp", offset, q3, offset)
        response = torch.exp(-0.5 * exponent.clamp_min(0))
        pixel_normals = F.normalize(-qd, dim=-1)
    else:
        response = torch.exp(-0.5 * squared)
        pixel_normals = normal[:, :, None].expand(-1, -1, rays.shape[1], -1)
        if mode == "radegs":
            depth = (
                torch.linalg.vector_norm(means, dim=-1)[:, :, None]
                + (gain[:, :, None] * delta).sum(-1)
            ) / torch.linalg.vector_norm(rays, dim=-1)[:, None]
        else:
            denominator = (pixel_normals * rays[:, None]).sum(-1)
            safe = torch.where(
                denominator.abs() > 1e-8, denominator, torch.ones_like(denominator)
            )
            depth = plane_distance[:, :, None] / safe
            response = torch.where(denominator.abs() > 1e-8, response, 0)
    response = torch.where(
        valid_slots[:, :, None]
        & (squared <= extent**2)
        & (depth > near)
        & (depth < far),
        response,
        0,
    )
    rgb, alpha, znum, nnum, l1, l2, median, weights = alpha_composite_batch(
        opacity[:, :, None] * response,
        color,
        depth,
        pixel_normals,
        distortions=distortions,
    )
    expected = znum / alpha.clamp_min(1e-8)
    if mode == "ambisur":
        dnum = (weights * plane_distance[:, :, None]).sum(1)
        denominator = (nnum * rays).sum(-1)
        safe = torch.where(
            denominator.abs() > 1e-8, denominator, torch.ones_like(denominator)
        )
        expected = torch.where(denominator.abs() > 1e-8, dnum / safe, 0)
    expected = torch.where(alpha > 1e-8, expected, 0)
    variance = (
        ray_color_variance_batch(weights, ray_color, rgb)
        if mode == "ambisur"
        else torch.zeros_like(alpha)
    )
    ambiguity = (weights * selection[:, :, None].to(weights.dtype)).sum(1)
    return torch.cat(
        (
            rgb + (1 - alpha[..., None]) * background,
            alpha[..., None],
            expected[..., None],
            F.normalize(nnum, dim=-1),
            l1[..., None],
            l2[..., None],
            median[..., None],
            variance[..., None],
            ambiguity[..., None],
        ),
        -1,
    )
