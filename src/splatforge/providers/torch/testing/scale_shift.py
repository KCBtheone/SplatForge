"""Torch provider for the non-public Phase 0 scale/shift fixture."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from ....core.digests import hash_path
from ....core.errors import DigestMismatchError, SchemaValidationError
from ....core.manifests import load_implementation_manifest
from ....core.models import (
    CallSignature,
    CapabilityRecord,
    OpDefinition,
    OpKey,
    SupportDecision,
)
from ....core.runtime import Runtime


IMPLEMENTATION_ID = "splatforge.torch.testing.scale_shift.v1"


def _finite_real(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaValidationError(f"{name} must be a real scalar")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise SchemaValidationError(f"{name} must be finite")
    return normalized


def validate_scale_shift_call(
    operands: tuple[Any, ...],
    attributes: Mapping[str, Any],
) -> Mapping[str, Any]:
    if len(operands) != 1:
        raise SchemaValidationError("scale_shift expects exactly one operand")
    value = operands[0]
    if not isinstance(value, torch.Tensor):
        raise SchemaValidationError("input must be a torch.Tensor")
    if not value.dtype.is_floating_point:
        raise SchemaValidationError("input must have a floating dtype")
    unexpected = sorted(set(attributes) - {"scale", "bias"})
    if unexpected:
        raise SchemaValidationError(
            "unexpected static attributes",
            details={"unexpected": unexpected},
        )
    return {
        "scale": _finite_real("scale", attributes.get("scale", 1.0)),
        "bias": _finite_real("bias", attributes.get("bias", 0.0)),
    }


def scale_shift(input: torch.Tensor, *, scale: float, bias: float) -> torch.Tensor:
    return input.mul(scale).add(bias)


def support_scale_shift(
    signature: CallSignature,
    _policy: Any,
) -> SupportDecision:
    if len(signature.tensor_operands) != 1:
        return SupportDecision.no("expected_one_tensor")
    return SupportDecision.yes()


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("Phase 0 fixture requires a SplatForge source checkout")


def create_phase0_runtime(repo_root: str | Path | None = None) -> Runtime:
    root = Path(repo_root).resolve() if repo_root is not None else _repo_root()
    manifest_path = Path(__file__).with_name("scale_shift.implementation.yaml")
    manifest = load_implementation_manifest(manifest_path)
    catalog = json.loads((root / "specs/catalog.json").read_text(encoding="utf-8"))
    catalog_entry = catalog["definitions"]["testing.scale_shift@1"]
    spec_path = root / catalog_entry["package"]
    actual_spec_revision = hash_path(spec_path)
    if actual_spec_revision != catalog_entry["spec_revision"]:
        raise DigestMismatchError(
            "Phase 0 OpSpec package does not match its catalog revision",
            details={
                "expected": catalog_entry["spec_revision"],
                "observed": actual_spec_revision,
            },
        )

    runtime = Runtime()
    runtime.register_definition(
        OpDefinition(
            op_key=OpKey.parse("testing.scale_shift@1"),
            spec_revision=catalog_entry["spec_revision"],
            validator=validate_scale_shift_call,
        )
    )
    runtime.register_implementation(
        manifest,
        scale_shift,
        support=support_scale_shift,
    )
    runtime.record_capability(
        CapabilityRecord(
            implementation_id=IMPLEMENTATION_ID,
            status="experimental",
            authority="phase0_fixture",
            environment_id="source-checkout",
            release_tier="best_effort",
        )
    )
    return runtime
