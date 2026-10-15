"""Segmented corrected-depth selection from the public paper equation."""

from __future__ import annotations

import torch
from torch import Tensor

from ._validation import (
    require_finite_scalar,
    require_same_device,
    require_same_length,
    require_sorted_segment_ids,
)
from .models import CorrectedDepthResult


def segmented_cumulative_sum(values: Tensor, segment_ids: Tensor) -> Tensor:
    """Inclusive cumulative sum for a nondecreasing segment-id vector."""

    if values.ndim != 1 or segment_ids.ndim != 1:
        raise ValueError("values and segment_ids must have shape [M]")
    require_same_length((("values", values), ("segment_ids", segment_ids)))
    require_same_device((("values", values), ("segment_ids", segment_ids)))
    require_sorted_segment_ids(segment_ids)
    if values.numel() == 0:
        return values.clone()
    cumulative = torch.cumsum(values, dim=0)
    starts_mask = torch.ones_like(segment_ids, dtype=torch.bool)
    starts_mask[1:] = segment_ids[1:] != segment_ids[:-1]
    starts = torch.nonzero(starts_mask, as_tuple=False).flatten()
    offsets = cumulative[starts] - values[starts]
    local_segment_index = torch.cumsum(starts_mask.to(torch.int64), dim=0) - 1
    return cumulative - offsets[local_segment_index]


def corrected_depth(
    depths: Tensor,
    opacities: Tensor,
    gaussian_weights: Tensor,
    segment_ids: Tensor,
    valid_mask: Tensor,
    *,
    num_segments: int,
    opacity_epsilon: float = 0.1,
    threshold: float = 0.6,
) -> CorrectedDepthResult:
    """Select the first ordered hit whose corrected accumulation exceeds a threshold.

    Invalid/no-crossing segments use a zero storage sentinel and are identified
    only by ``valid_mask=False``.  Threshold comparison is strict (``>``), as in
    the paper.  Hit order is never changed when actual ray depths locally invert.
    """

    opacity_epsilon = require_finite_scalar("opacity_epsilon", opacity_epsilon)
    threshold = require_finite_scalar("threshold", threshold)
    if opacity_epsilon < 0.0:
        raise ValueError("opacity_epsilon must be nonnegative")
    if threshold < 0.0:
        raise ValueError("threshold must be nonnegative")
    if num_segments <= 0:
        raise ValueError("num_segments must be positive")
    if any(item.ndim != 1 for item in (depths, opacities, gaussian_weights, valid_mask)):
        raise ValueError("depths, opacities, gaussian_weights, and valid_mask must be [M]")
    if not depths.dtype.is_floating_point:
        raise TypeError("depths must have a floating dtype")
    if opacities.dtype != depths.dtype or gaussian_weights.dtype != depths.dtype:
        raise TypeError("depths, opacities, and gaussian_weights must share dtype")
    if valid_mask.dtype != torch.bool:
        raise TypeError("valid_mask must be boolean")
    require_same_length(
        (
            ("depths", depths),
            ("opacities", opacities),
            ("gaussian_weights", gaussian_weights),
            ("segment_ids", segment_ids),
            ("valid_mask", valid_mask),
        )
    )
    require_same_device(
        (
            ("depths", depths),
            ("opacities", opacities),
            ("gaussian_weights", gaussian_weights),
            ("segment_ids", segment_ids),
            ("valid_mask", valid_mask),
        )
    )
    require_sorted_segment_ids(segment_ids, num_segments=num_segments)

    if depths.is_cuda:
        from ._cuda import corrected_depth_cuda, cuda_available

        if cuda_available():
            dense_depth, dense_valid, selected, contribution, cumulative = corrected_depth_cuda(
                depths,
                opacities,
                gaussian_weights,
                segment_ids.to(torch.int64),
                valid_mask,
                num_segments,
                opacity_epsilon,
                threshold,
            )
            return CorrectedDepthResult(
                depth=dense_depth,
                valid_mask=dense_valid,
                selected_hit_indices=selected,
                contribution=contribution,
                cumulative_contribution=cumulative,
            )

    raw_contribution = (opacities + opacity_epsilon) * gaussian_weights
    keep = valid_mask & torch.isfinite(raw_contribution)
    contribution = torch.where(keep, raw_contribution, raw_contribution.new_zeros(()))
    cumulative = segmented_cumulative_sum(contribution, segment_ids)
    if depths.numel() == 0:
        return CorrectedDepthResult(
            depth=depths.new_zeros((num_segments,)),
            valid_mask=torch.zeros(num_segments, dtype=torch.bool, device=depths.device),
            selected_hit_indices=torch.full(
                (num_segments,), -1, dtype=torch.int64, device=depths.device
            ),
            contribution=contribution,
            cumulative_contribution=cumulative,
        )

    crossing = valid_mask & torch.isfinite(cumulative) & (cumulative > threshold)
    sentinel = int(depths.numel())
    candidates = torch.arange(depths.numel(), dtype=torch.int64, device=depths.device)
    candidates.masked_fill_(~crossing, sentinel)
    selected = torch.full(
        (num_segments,), sentinel, dtype=torch.int64, device=depths.device
    )
    selected.scatter_reduce_(
        0, segment_ids.to(torch.int64), candidates, reduce="amin", include_self=True
    )
    dense_valid = selected < sentinel
    safe_selected = selected.clamp(max=sentinel - 1)
    dense_depth = torch.where(dense_valid, depths[safe_selected], depths.new_zeros(()))
    selected = torch.where(dense_valid, selected, selected.new_full((), -1))
    return CorrectedDepthResult(
        depth=dense_depth,
        valid_mask=dense_valid,
        selected_hit_indices=selected,
        contribution=contribution,
        cumulative_contribution=cumulative,
    )
