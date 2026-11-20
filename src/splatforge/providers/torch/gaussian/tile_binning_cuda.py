"""Exact GPU CSR footprint lists with stable depth/index order.

Discrete operation, explicitly detaches bounds. Two scalar synchronizations
validate finiteness and size a variable-length output. No N-sized host copy.
Not CUDA-graph capturable. No Gaussian/pair truncation or CPU fallback.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import torch

_extension = None


def extension():
    global _extension
    if _extension is None:
        from torch.utils.cpp_extension import load

        root = os.environ.get("TORCH_EXTENSIONS_DIR")
        if not root:
            raise RuntimeError("Set TORCH_EXTENSIONS_DIR before loading the tile binning CUDA extension")
        source = Path(__file__).with_suffix(".cu")
        from ...._paths import output_path

        build = output_path(Path(root).expanduser() / "tile_binning_v1")
        build.mkdir(parents=True, exist_ok=True)
        _extension = load(
            name="sf_tile_binning_v1",
            sources=[str(source)],
            build_directory=str(build),
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--fmad=false"],
            verbose=False,
        )
    return _extension


def tile_members_cuda(bounds, *, width, height, tile_size, near, far):
    if (
        bounds.device.type != "cuda"
        or bounds.dtype != torch.float32
        or not bounds.is_contiguous()
    ):
        raise ValueError("explicit contiguous CUDA float32 bounds required")
    if (
        min(width, height, tile_size) <= 0
        or not math.isfinite(near)
        or not math.isfinite(far)
        or not near < far
    ):
        raise ValueError("invalid binning settings")
    bounds = bounds.detach()
    tiles = math.ceil(width / tile_size) * math.ceil(height / tile_size)
    counts, error = extension().count(bounds, width, height, tile_size, near, far)
    if error.item():
        raise FloatingPointError("non-finite projected bounds")
    ends = counts.cumsum(0)
    total = int(ends[-1].item()) if len(ends) else 0
    if total >= 2**31:
        raise ValueError("pair count exceeds explicit 32-bit implementation limit")
    order = bounds[:, 3].argsort(stable=True)
    keys, ids = extension().emit(
        bounds, order, ends - counts, total, width, height, tile_size, near, far
    )
    keys, permutation = keys.sort()
    ids = ids[permutation]
    lengths = (
        torch.bincount(keys // len(bounds), minlength=tiles)
        if len(bounds)
        else torch.zeros(tiles, device=bounds.device, dtype=torch.int64)
    )
    offsets = torch.cat((lengths.new_zeros(1), lengths.cumsum(0)))
    return offsets, ids
