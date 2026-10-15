"""Loss assembly utilities for corrected-depth normal supervision."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from ._validation import (
    require_same_device,
    require_same_length,
    require_sorted_segment_ids,
)
from .models import DenseNormalConsistencyResult, NormalConsistencyResult


NormalReduction = Literal["sum", "mean_hits", "mean_segments", "mean_active_segments"]
DenseNormalReduction = Literal["sum", "mean_segments", "mean_active_segments"]


def segmented_exclusive_transmittance(alpha: Tensor, segment_ids: Tensor) -> Tensor:
    """Compute per-hit exclusive transmittance for sorted ray segments."""

    if alpha.ndim != 1 or segment_ids.ndim != 1:
        raise ValueError("alpha and segment_ids must have shape [M]")
    if not alpha.dtype.is_floating_point:
        raise TypeError("alpha must have a floating dtype")
    require_same_length((("alpha", alpha), ("segment_ids", segment_ids)))
    require_same_device((("alpha", alpha), ("segment_ids", segment_ids)))
    require_sorted_segment_ids(segment_ids)
    if alpha.numel() == 0:
        return alpha.clone()
    if bool(torch.any(~torch.isfinite(alpha)).item()):
        raise ValueError("alpha must be finite")
    if bool(torch.any((alpha < 0.0) | (alpha >= 1.0)).item()):
        raise ValueError("alpha must satisfy 0 <= alpha < 1")

    log_survival = torch.log1p(-alpha)
    inclusive = torch.cumsum(log_survival, dim=0)
    starts_mask = torch.ones_like(segment_ids, dtype=torch.bool)
    starts_mask[1:] = segment_ids[1:] != segment_ids[:-1]
    starts = torch.nonzero(starts_mask, as_tuple=False).flatten()
    offsets = inclusive[starts] - log_survival[starts]
    local_segment_index = torch.cumsum(starts_mask.to(torch.int64), dim=0) - 1
    exclusive_log = inclusive - log_survival - offsets[local_segment_index]
    return torch.exp(exclusive_log)


def blending_weights(
    opacities: Tensor,
    gaussian_weights: Tensor,
    segment_ids: Tensor,
    valid_mask: Tensor,
) -> Tensor:
    """Recreate gsplat's front-to-back alpha blending weights for accepted hits."""

    if opacities.ndim != 1 or gaussian_weights.ndim != 1 or valid_mask.ndim != 1:
        raise ValueError("opacities, gaussian_weights, and valid_mask must have shape [M]")
    if opacities.dtype != gaussian_weights.dtype or not opacities.dtype.is_floating_point:
        raise TypeError("opacities and gaussian_weights must share a floating dtype")
    if valid_mask.dtype != torch.bool:
        raise TypeError("valid_mask must be boolean")
    require_same_length(
        (
            ("opacities", opacities),
            ("gaussian_weights", gaussian_weights),
            ("segment_ids", segment_ids),
            ("valid_mask", valid_mask),
        )
    )
    require_same_device(
        (
            ("opacities", opacities),
            ("gaussian_weights", gaussian_weights),
            ("segment_ids", segment_ids),
            ("valid_mask", valid_mask),
        )
    )
    require_sorted_segment_ids(segment_ids)
    effective_alpha = torch.clamp_max(opacities * gaussian_weights, 0.999)
    effective_alpha = torch.where(valid_mask, effective_alpha, torch.zeros_like(effective_alpha))
    transmittance = segmented_exclusive_transmittance(effective_alpha, segment_ids)
    return effective_alpha * transmittance


def _normal_reduction(
    numerator: Tensor,
    hit_mask: Tensor,
    hit_segments: Tensor,
    *,
    reduction: NormalReduction,
    num_segments: int,
) -> tuple[Tensor, Tensor]:
    if reduction == "sum":
        denominator = torch.ones((), dtype=numerator.dtype, device=numerator.device)
    elif reduction == "mean_hits":
        denominator = hit_mask.sum().to(numerator.dtype)
    elif reduction == "mean_segments":
        denominator = torch.tensor(
            num_segments, dtype=numerator.dtype, device=numerator.device
        )
    elif reduction == "mean_active_segments":
        active = torch.zeros(num_segments, dtype=torch.bool, device=numerator.device)
        if hit_segments.numel():
            active.scatter_(0, hit_segments[hit_mask], True)
        denominator = active.sum().to(numerator.dtype)
    else:
        raise ValueError(f"unsupported reduction: {reduction!r}")
    return numerator / denominator.clamp_min(1.0), denominator


def normal_consistency_loss(
    surfel_normals: Tensor,
    surface_normals: Tensor,
    surface_valid_mask: Tensor,
    opacities: Tensor,
    gaussian_weights: Tensor,
    segment_ids: Tensor,
    hit_valid_mask: Tensor,
    *,
    num_segments: int,
    reduction: NormalReduction,
    normalization_epsilon: float = 1e-12,
) -> NormalConsistencyResult:
    """Evaluate ``sum_i blend_weight_i * (1 - n_i dot N)``.

    ``surfel_normals`` and ``surface_normals`` must already use the same frame
    and orientation convention.  The function normalizes both inputs and
    excludes zero/non-finite normal vectors rather than inventing directions.
    """

    if num_segments <= 0:
        raise ValueError("num_segments must be positive")
    if normalization_epsilon < 0.0:
        raise ValueError("normalization_epsilon must be nonnegative")
    if surfel_normals.ndim != 2 or surfel_normals.shape[-1] != 3:
        raise ValueError("surfel_normals must have shape [M, 3]")
    if surface_normals.shape != (num_segments, 3):
        raise ValueError("surface_normals must have shape [num_segments, 3]")
    if surface_valid_mask.shape != (num_segments,) or surface_valid_mask.dtype != torch.bool:
        raise ValueError("surface_valid_mask must be boolean [num_segments]")
    if hit_valid_mask.dtype != torch.bool:
        raise TypeError("hit_valid_mask must be boolean")
    require_same_length(
        (
            ("surfel_normals", surfel_normals),
            ("opacities", opacities),
            ("gaussian_weights", gaussian_weights),
            ("segment_ids", segment_ids),
            ("hit_valid_mask", hit_valid_mask),
        )
    )
    require_same_device(
        (
            ("surfel_normals", surfel_normals),
            ("surface_normals", surface_normals),
            ("surface_valid_mask", surface_valid_mask),
            ("opacities", opacities),
            ("gaussian_weights", gaussian_weights),
            ("segment_ids", segment_ids),
            ("hit_valid_mask", hit_valid_mask),
        )
    )
    require_sorted_segment_ids(segment_ids, num_segments=num_segments)
    weights = blending_weights(
        opacities, gaussian_weights, segment_ids, hit_valid_mask
    )
    if surfel_normals.numel() == 0:
        zero = surfel_normals.sum() * 0.0
        empty_mask = torch.empty(0, dtype=torch.bool, device=surfel_normals.device)
        loss, denominator = _normal_reduction(
            zero,
            empty_mask,
            segment_ids,
            reduction=reduction,
            num_segments=num_segments,
        )
        return NormalConsistencyResult(
            loss=loss,
            numerator=zero,
            denominator=denominator,
            per_hit=weights,
            blending_weights=weights,
            hit_mask=empty_mask,
            per_segment=surfel_normals.new_zeros((num_segments,)),
        )

    surfel_magnitude = torch.linalg.vector_norm(surfel_normals, dim=-1)
    surface_magnitude = torch.linalg.vector_norm(surface_normals, dim=-1)
    surfel_ok = torch.isfinite(surfel_normals).all(dim=-1) & (
        surfel_magnitude > normalization_epsilon
    )
    surface_ok = (
        surface_valid_mask
        & torch.isfinite(surface_normals).all(dim=-1)
        & (surface_magnitude > normalization_epsilon)
    )
    normalized_surfel = F.normalize(
        torch.where(surfel_ok[:, None], surfel_normals, torch.zeros_like(surfel_normals)),
        dim=-1,
        eps=max(normalization_epsilon, torch.finfo(surfel_normals.dtype).tiny),
    )
    normalized_surface = F.normalize(
        torch.where(surface_ok[:, None], surface_normals, torch.zeros_like(surface_normals)),
        dim=-1,
        eps=max(normalization_epsilon, torch.finfo(surface_normals.dtype).tiny),
    )
    per_hit_surface = normalized_surface[segment_ids]
    dot = torch.sum(normalized_surfel * per_hit_surface, dim=-1)
    hit_mask = hit_valid_mask & surfel_ok & surface_ok[segment_ids]
    per_hit = torch.where(hit_mask, weights * (1.0 - dot), torch.zeros_like(weights))
    numerator = per_hit.sum()
    per_segment = surfel_normals.new_zeros((num_segments,))
    per_segment.scatter_add_(0, segment_ids, per_hit)
    loss, denominator = _normal_reduction(
        numerator,
        hit_mask,
        segment_ids,
        reduction=reduction,
        num_segments=num_segments,
    )
    return NormalConsistencyResult(
        loss=loss,
        numerator=numerator,
        denominator=denominator,
        per_hit=per_hit,
        blending_weights=weights,
        hit_mask=hit_mask,
        per_segment=per_segment,
    )


def normal_consistency_from_rendered(
    rendered_normals: Tensor,
    rendered_alphas: Tensor,
    surface_normals: Tensor,
    surface_valid_mask: Tensor,
    *,
    reduction: DenseNormalReduction,
    normalization_epsilon: float = 1e-12,
) -> DenseNormalConsistencyResult:
    """Use primary gsplat accumulations to evaluate the hit-weighted normal loss.

    Since gsplat returns ``sum_i(weight_i * n_i)`` and ``sum_i(weight_i)``,
    ``alpha - dot(rendered_normal, N)`` is algebraically identical to the
    paper's per-hit sum while reusing the rasterizer's exact accepted set,
    cutoff, transmittance, and numerical order.
    """

    if rendered_normals.shape[-1:] != (3,) or surface_normals.shape != rendered_normals.shape:
        raise ValueError("rendered_normals and surface_normals must share shape [..., 3]")
    if rendered_alphas.shape == rendered_normals.shape[:-1] + (1,):
        rendered_alphas = rendered_alphas[..., 0]
    if rendered_alphas.shape != rendered_normals.shape[:-1]:
        raise ValueError("rendered_alphas must have shape [...] or [..., 1]")
    if surface_valid_mask.shape != rendered_normals.shape[:-1]:
        raise ValueError("surface_valid_mask must match the normal image")
    if surface_valid_mask.dtype != torch.bool:
        raise TypeError("surface_valid_mask must be boolean")
    if not rendered_normals.dtype.is_floating_point:
        raise TypeError("normal tensors must have floating dtype")
    if rendered_normals.dtype != surface_normals.dtype or rendered_normals.dtype != rendered_alphas.dtype:
        raise TypeError("rendered outputs and surface normals must share dtype")
    require_same_device(
        (
            ("rendered_normals", rendered_normals),
            ("rendered_alphas", rendered_alphas),
            ("surface_normals", surface_normals),
            ("surface_valid_mask", surface_valid_mask),
        )
    )
    if normalization_epsilon < 0.0:
        raise ValueError("normalization_epsilon must be nonnegative")

    surface_magnitude = torch.linalg.vector_norm(surface_normals, dim=-1)
    valid = (
        surface_valid_mask
        & torch.isfinite(surface_normals).all(dim=-1)
        & torch.isfinite(rendered_normals).all(dim=-1)
        & torch.isfinite(rendered_alphas)
        & (surface_magnitude > normalization_epsilon)
    )
    normalized_surface = F.normalize(
        torch.where(valid[..., None], surface_normals, torch.zeros_like(surface_normals)),
        dim=-1,
        eps=max(normalization_epsilon, torch.finfo(surface_normals.dtype).tiny),
    )
    raw = rendered_alphas - torch.sum(rendered_normals * normalized_surface, dim=-1)
    per_segment = torch.where(valid, raw, torch.zeros_like(raw)).reshape(-1)
    dense_valid = valid.reshape(-1)
    numerator = per_segment.sum()
    if reduction == "sum":
        denominator = torch.ones((), dtype=numerator.dtype, device=numerator.device)
    elif reduction == "mean_segments":
        denominator = torch.tensor(
            per_segment.numel(), dtype=numerator.dtype, device=numerator.device
        )
    elif reduction == "mean_active_segments":
        denominator = dense_valid.sum().to(numerator.dtype)
    else:
        raise ValueError(f"unsupported dense normal reduction: {reduction!r}")
    return DenseNormalConsistencyResult(
        loss=numerator / denominator.clamp_min(1.0),
        numerator=numerator,
        denominator=denominator,
        per_segment=per_segment,
        valid_mask=dense_valid,
    )
