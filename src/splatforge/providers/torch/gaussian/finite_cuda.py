"""Explicit CUDA float32 finite predicate; one flag, no operand copies/casts.

Training safety utility, not a differentiable surface-method operator. The
caller explicitly synchronizes the returned device int32 scalar if required.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

_extension = None


def cuda_all_finite(tensors):
    global _extension
    if _extension is None:
        from torch.utils.cpp_extension import load

        directory = os.environ.get("TORCH_EXTENSIONS_DIR")
        if not directory:
            raise RuntimeError("Set TORCH_EXTENSIONS_DIR before loading the finite-check CUDA extension")
        source = Path(__file__).with_suffix(".cu")
        token = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
        from ...._paths import output_path

        build = output_path(Path(directory).expanduser() / ("finite_" + token))
        build.mkdir(parents=True, exist_ok=True)
        _extension = load(
            name="sf_finite_" + token,
            sources=[str(source)],
            build_directory=str(build),
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--fmad=false"],
            verbose=False,
        )
    return _extension.all_finite(list(tensors))
