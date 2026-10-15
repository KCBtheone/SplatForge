"""Segmented Unbiased Surfel depth-convergence loss."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

from ._validation import (
    require_finite_scalar,
    require_same_device,
    require_same_length,
    require_sorted_segment_ids,
)
from .models import ConvergenceResult


Reduction = Literal["sum", "mean_pairs", "mean_segments", "mean_active_segments"]


class _AsymmetricSquaredDifference(torch.autograd.Function):
    """Squared difference with the paper's non-conservative far-depth VJP."""

    @staticmethod
    def forward(ctx, previous: Tensor, current: Tensor, far_gradient_scale: float) -> Tensor:
        delta = current - previous
        ctx.save_for_backward(delta)
        ctx.far_gradient_scale = float(far_gradient_scale)
        return delta.square()

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, Tensor, None]:
        (delta,) = ctx.saved_tensors
        common = 2.0 * delta * grad_output
        return -common, common * ctx.far_gradient_scale, None


def asymmetric_squared_difference(
    previous: Tensor, current: Tensor, *, far_gradient_scale: float = 1.25
) -> Tensor:
    """Apply the published first-order VJP; higher-order gradients are unsupported."""

    far_gradient_scale = require_finite_scalar(
        "far_gradient_scale", far_gradient_scale, positive=True
    )
    return _AsymmetricSquaredDifference.apply(previous, current, far_gradient_scale)


def segmented_adjacent_pairs(segment_ids: Tensor) -> tuple[Tensor, Tensor]:
    """Return adjacent indices only where both hits belong to the same ray."""

    require_sorted_segment_ids(segment_ids)
    if segment_ids.numel() < 2:
        empty = torch.empty(0, dtype=torch.int64, device=segment_ids.device)
        return empty, empty
    current = torch.arange(1, segment_ids.numel(), device=segment_ids.device)
    same = segment_ids[1:] == segment_ids[:-1]
    current = current[same]
    return current - 1, current


def _reduce(
    numerator: Tensor,
    pair_mask: Tensor,
    pair_segments: Tensor,
    *,
    reduction: Reduction,
    num_segments: int,
) -> tuple[Tensor, Tensor]:
    if reduction == "sum":
        denominator = torch.ones((), dtype=numerator.dtype, device=numerator.device)
    elif reduction == "mean_pairs":
        denominator = pair_mask.sum().to(numerator.dtype)
    elif reduction == "mean_segments":
        denominator = torch.tensor(
            num_segments, dtype=numerator.dtype, device=numerator.device
        )
    elif reduction == "mean_active_segments":
        active = torch.zeros(num_segments, dtype=torch.bool, device=numerator.device)
        if pair_segments.numel():
            active.scatter_(0, pair_segments[pair_mask], True)
        denominator = active.sum().to(numerator.dtype)
    else:
        raise ValueError(f"unsupported reduction: {reduction!r}")
    return numerator / denominator.clamp_min(1.0), denominator


def depth_convergence(
    depths: Tensor,
    gaussian_weights: Tensor,
    segment_ids: Tensor,
    valid_mask: Tensor,
    *,
    scene_radius: float,
    num_segments: int,
    reduction: Reduction,
    far_gradient_scale: float = 1.25,
) -> ConvergenceResult:
    """Compute the adjacent-hit convergence objective and explicit normalizer.

    The Gaussian weight is detached exactly as required by the paper.  Opacity
    is intentionally not an input.  Pairs with ``abs(delta_depth)`` strictly
    greater than ``scene_radius / 4`` are excluded; equality is included.
    """

    scene_radius = require_finite_scalar("scene_radius", scene_radius, positive=True)
    far_gradient_scale = require_finite_scalar(
        "far_gradient_scale", far_gradient_scale, positive=True
    )
    if num_segments <= 0:
        raise ValueError("num_segments must be positive")
    if depths.ndim != 1 or gaussian_weights.ndim != 1 or valid_mask.ndim != 1:
        raise ValueError("depths, gaussian_weights, and valid_mask must have shape [M]")
    if not depths.dtype.is_floating_point or not gaussian_weights.dtype.is_floating_point:
        raise TypeError("depths and gaussian_weights must be floating tensors")
    if depths.dtype != gaussian_weights.dtype:
        raise TypeError("depths and gaussian_weights must have the same dtype")
    if valid_mask.dtype != torch.bool:
        raise TypeError("valid_mask must be boolean")
    require_same_length(
        (
            ("depths", depths),
            ("gaussian_weights", gaussian_weights),
            ("segment_ids", segment_ids),
            ("valid_mask", valid_mask),
        )
    )
    require_same_device(
        (
            ("depths", depths),
            ("gaussian_weights", gaussian_weights),
            ("segment_ids", segment_ids),
            ("valid_mask", valid_mask),
        )
    )
    require_sorted_segment_ids(segment_ids, num_segments=num_segments)

    if depths.is_cuda:
        from ._cuda import convergence_pairs_cuda, cuda_available

        if cuda_available() and depths.numel() >= 2:
            per_pair_dense, pair_mask_dense, prev_dense, curr_dense = convergence_pairs_cuda(
                depths,
                gaussian_weights,
                segment_ids.to(torch.int64),
                valid_mask,
                scene_radius,
                far_gradient_scale,
            )
            keep = pair_mask_dense
            previous = prev_dense[keep]
            current = curr_dense[keep]
            pair_mask = pair_mask_dense[keep]
            per_pair = per_pair_dense[keep]
            numerator = per_pair.sum()
            pair_segments = segment_ids.to(torch.int64)[current]
            per_segment = depths.new_zeros((num_segments,))
            if pair_segments.numel():
                per_segment.scatter_add_(0, pair_segments, per_pair)
            loss, denominator = _reduce(
                numerator, pair_mask, pair_segments, reduction=reduction, num_segments=num_segments
            )
            return ConvergenceResult(
                loss=loss,
                numerator=numerator,
                denominator=denominator,
                per_pair=per_pair,
                pair_mask=pair_mask,
                previous_indices=previous,
                current_indices=current,
                per_segment=per_segment,
            )

    previous, current = segmented_adjacent_pairs(segment_ids)
    if current.numel() == 0:
        empty = depths.new_empty((0,))
        empty_mask = torch.empty(0, dtype=torch.bool, device=depths.device)
        per_segment = depths.new_zeros((num_segments,))
        zero = depths.sum() * 0.0
        loss, denominator = _reduce(
            zero,
            empty_mask,
            segment_ids.new_empty((0,)),
            reduction=reduction,
            num_segments=num_segments,
        )
        return ConvergenceResult(
            loss=loss,
            numerator=zero,
            denominator=denominator,
            per_pair=empty,
            pair_mask=empty_mask,
            previous_indices=previous,
            current_indices=current,
            per_segment=per_segment,
        )

    delta = depths[current] - depths[previous]
    pair_mask = valid_mask[current] & valid_mask[previous]
    pair_mask = pair_mask & torch.isfinite(delta)
    pair_mask = pair_mask & (delta.abs() <= scene_radius / 4.0)
    detached_weight = torch.minimum(
        gaussian_weights[current], gaussian_weights[previous]
    ).detach()
    squared = asymmetric_squared_difference(
        depths[previous], depths[current], far_gradient_scale=far_gradient_scale
    )
    per_pair = torch.where(pair_mask, detached_weight * squared, torch.zeros_like(squared))
    numerator = per_pair.sum()
    pair_segments = segment_ids[current]
    per_segment = depths.new_zeros((num_segments,))
    per_segment.scatter_add_(0, pair_segments, per_pair)
    loss, denominator = _reduce(
        numerator,
        pair_mask,
        pair_segments,
        reduction=reduction,
        num_segments=num_segments,
    )
    return ConvergenceResult(
        loss=loss,
        numerator=numerator,
        denominator=denominator,
        per_pair=per_pair,
        pair_mask=pair_mask,
        previous_indices=previous,
        current_indices=current,
        per_segment=per_segment,
    )
