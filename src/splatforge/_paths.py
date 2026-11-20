"""Storage boundary for run outputs and extension builds."""

import os
from pathlib import Path


def data_root() -> Path | None:
    configured = os.environ.get("SPLATFORGE_DATA_ROOT")
    return Path(configured).expanduser().resolve() if configured else None


def output_path(path) -> Path:
    resolved = Path(path).expanduser().resolve()
    root = data_root()
    if root is not None and not resolved.is_relative_to(root):
        raise ValueError(f"Output or build path must be inside SPLATFORGE_DATA_ROOT: {resolved}")
    return resolved
