"""Executable virtual-operator lifecycle used as the Phase 0 exit smoke."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch

from ..core.digests import hash_file, hash_path
from ..core.policy import ExecutionPolicy
from ..providers.torch.testing.scale_shift import (
    IMPLEMENTATION_ID,
    create_phase0_runtime,
)
from .conformance import ConformanceCase, run_conformance


def _revision(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if completed.returncode == 0:
        return completed.stdout.strip()
    return "unversioned"


def build_report(root: Path, output: Path) -> dict[str, object]:
    runtime = create_phase0_runtime(root)
    policy = ExecutionPolicy(
        providers=("torch",),
        implementation_id=IMPLEMENTATION_ID,
        evidence_requirement="any",
        allow_unverified=True,
        copies="forbid",
        host_sync="forbid",
    )
    cases = (
        ConformanceCase(
            name="basic_float32",
            operands=(torch.tensor([-2.0, 0.0, 3.0], dtype=torch.float32),),
            attributes={"scale": 2.0, "bias": 1.0},
            expected=torch.tensor([-3.0, 1.0, 7.0], dtype=torch.float32),
        ),
        ConformanceCase(
            name="empty_float64",
            operands=(torch.empty((0, 4), dtype=torch.float64),),
            attributes={"scale": -3.0, "bias": 0.25},
            expected=torch.empty((0, 4), dtype=torch.float64),
        ),
        ConformanceCase(
            name="default_attributes",
            operands=(torch.tensor([1.5], dtype=torch.float64),),
            attributes={},
            expected=torch.tensor([1.5], dtype=torch.float64),
        ),
    )
    provider_source = root / "src/splatforge/providers/torch/testing/scale_shift.py"
    conformance_path = root / "tests/conformance/testing"
    return run_conformance(
        runtime,
        "testing.scale_shift@1",
        cases,
        policy=policy,
        repository_revision=_revision(root),
        artifact_digest=hash_file(provider_source),
        conformance_digest=hash_file(conformance_path / "test_scale_shift.py"),
        output_path=output,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/phase0/phase0_conformance.local.json"),
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else args.repo / args.output
    report = build_report(args.repo.resolve(), output.resolve())
    print(json.dumps(report["summary"], sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
