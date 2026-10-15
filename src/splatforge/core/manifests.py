"""Safe manifest loading for trusted, already-validated local files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .models import ImplementationManifest


def load_yaml(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"manifest root must be a mapping: {manifest_path}")
    return value


def load_implementation_manifest(path: str | Path) -> ImplementationManifest:
    return ImplementationManifest.from_dict(load_yaml(path))

