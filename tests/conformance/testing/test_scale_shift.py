from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import jsonschema
import torch

from splatforge.core.digests import hash_file, hash_path
from splatforge.core.policy import ExecutionPolicy
from splatforge.providers.torch.testing.scale_shift import (
    IMPLEMENTATION_ID,
    create_phase0_runtime,
)
from splatforge.testing.conformance import ConformanceCase, run_conformance


class ScaleShiftConformanceTests(unittest.TestCase):
    def test_exact_provider_emits_schema_valid_report(self) -> None:
        root = Path(__file__).resolve().parents[3]
        runtime = create_phase0_runtime(root)
        policy = ExecutionPolicy(
            providers=("torch",),
            implementation_id=IMPLEMENTATION_ID,
            evidence_requirement="any",
            allow_unverified=True,
        )
        cases = (
            ConformanceCase(
                name="hand_computed",
                operands=(torch.tensor([-1.0, 2.0], dtype=torch.float64),),
                attributes={"scale": 3.0, "bias": -2.0},
                expected=torch.tensor([-5.0, 4.0], dtype=torch.float64),
            ),
        )
        with tempfile.TemporaryDirectory() as raw:
            report_path = Path(raw) / "report.json"
            report = run_conformance(
                runtime,
                "testing.scale_shift@1",
                cases,
                policy=policy,
                repository_revision="unversioned-test",
                artifact_digest=hash_file(
                    root / "src/splatforge/providers/torch/testing/scale_shift.py"
                ),
                conformance_digest=hash_file(
                    root / "tests/conformance/testing/test_scale_shift.py"
                ),
                output_path=report_path,
            )
            self.assertEqual(report["status"], "passed")
            persisted = json.loads(report_path.read_text(encoding="utf-8"))
            schema = json.loads(
                (root / "tools/schemas/evidence_report.schema.json").read_text(
                    encoding="utf-8"
                )
            )
            jsonschema.Draft202012Validator(
                schema,
                format_checker=jsonschema.FormatChecker(),
            ).validate(persisted)

    def test_gradient_is_exact_for_input(self) -> None:
        runtime = create_phase0_runtime()
        policy = ExecutionPolicy(
            providers=("torch",),
            implementation_id=IMPLEMENTATION_ID,
            evidence_requirement="any",
            allow_unverified=True,
        )
        value = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
        output = runtime.execute(
            "testing.scale_shift@1",
            value,
            policy=policy,
            scale=-4.0,
            bias=3.0,
        )
        output.sum().backward()
        torch.testing.assert_close(value.grad, torch.full_like(value, -4.0))


if __name__ == "__main__":
    unittest.main()
