"""Small validation helpers shared by the pure PyTorch operators."""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import Tensor


def require_tensor(name: str, value: object, *, ndim: int | None = None) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}, got shape {tuple(value.shape)}")
    return value


def require_floating(name: str, value: Tensor) -> None:
    if not value.dtype.is_floating_point:
        raise TypeError(f"{name} must have a floating dtype, got {value.dtype}")


def require_same_length(named_tensors: Iterable[tuple[str, Tensor]]) -> int:
    items = tuple(named_tensors)
    if not items:
        raise ValueError("at least one tensor is required")
    length = int(items[0][1].shape[0])
    mismatches = [
        f"{name}={tuple(tensor.shape)}"
        for name, tensor in items
        if tensor.ndim == 0 or int(tensor.shape[0]) != length
    ]
    if mismatches:
        raise ValueError(
            f"all tensors must share leading length {length}; " + ", ".join(mismatches)
        )
    return length


def require_same_device(named_tensors: Iterable[tuple[str, Tensor]]) -> torch.device:
    items = tuple(named_tensors)
    if not items:
        raise ValueError("at least one tensor is required")
    device = items[0][1].device
    mismatches = [f"{name}={tensor.device}" for name, tensor in items if tensor.device != device]
    if mismatches:
        raise ValueError(
            f"all tensors must be on {device}; mismatched devices: " + ", ".join(mismatches)
        )
    return device


def require_nonnegative_integer_ids(name: str, ids: Tensor) -> None:
    if ids.ndim != 1:
        raise ValueError(f"{name} must have shape [M], got {tuple(ids.shape)}")
    if ids.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise TypeError(f"{name} must have an integer dtype, got {ids.dtype}")
    if ids.numel() and bool(torch.any(ids < 0).item()):
        raise ValueError(f"{name} must contain only nonnegative ids")


def require_sorted_segment_ids(segment_ids: Tensor, *, num_segments: int | None = None) -> None:
    require_nonnegative_integer_ids("segment_ids", segment_ids)
    if segment_ids.numel() > 1 and bool(torch.any(segment_ids[1:] < segment_ids[:-1]).item()):
        raise ValueError("segment_ids must be nondecreasing; hit order must not be silently sorted")
    if num_segments is not None:
        if num_segments < 0:
            raise ValueError("num_segments must be nonnegative")
        if segment_ids.numel() and int(segment_ids[-1].item()) >= num_segments:
            raise ValueError("segment_ids contains an id outside num_segments")


def require_finite_scalar(name: str, value: float, *, positive: bool = False) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    if positive and normalized <= 0.0:
        raise ValueError(f"{name} must be positive")
    return normalized
