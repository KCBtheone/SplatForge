from __future__ import annotations

import unittest
from dataclasses import replace

import torch

from splatforge.core.errors import DispatchError, PlanSignatureMismatchError
from splatforge.core.policy import ExecutionPolicy
from splatforge.core.models import CapabilityRecord
from splatforge.providers.torch.testing.scale_shift import (
    IMPLEMENTATION_ID,
    create_phase0_runtime,
)


def exact_policy() -> ExecutionPolicy:
    return ExecutionPolicy(
        providers=("torch",),
        implementation_id=IMPLEMENTATION_ID,
        evidence_requirement="any",
        allow_unverified=True,
        copies="forbid",
        host_sync="forbid",
    )


class RuntimeTests(unittest.TestCase):
    def test_exact_implementation_is_selected_and_executed(self) -> None:
        runtime = create_phase0_runtime()
        value = torch.tensor([-2.0, 0.0, 3.0], dtype=torch.float32)
        explanation = runtime.explain_dispatch(
            "testing.scale_shift@1",
            value,
            policy=exact_policy(),
            scale=2.0,
            bias=1.0,
        )
        self.assertEqual(explanation.selected_implementation_id, IMPLEMENTATION_ID)
        self.assertTrue(explanation.candidates[0].supported)
        actual = runtime.execute(
            "testing.scale_shift@1",
            value,
            policy=exact_policy(),
            scale=2.0,
            bias=1.0,
        )
        torch.testing.assert_close(
            actual,
            torch.tensor([-3.0, 1.0, 7.0], dtype=torch.float32),
        )

    def test_default_release_policy_rejects_experimental_fixture(self) -> None:
        runtime = create_phase0_runtime()
        value = torch.tensor([1.0], dtype=torch.float32)
        explanation = runtime.explain_dispatch("testing.scale_shift@1", value)
        self.assertIsNone(explanation.selected_implementation_id)
        self.assertIn("release_evidence_required", explanation.candidates[0].reasons)
        with self.assertRaises(DispatchError):
            runtime.execute("testing.scale_shift@1", value)

    def test_fixed_provider_never_falls_back_to_torch(self) -> None:
        runtime = create_phase0_runtime()
        value = torch.tensor([1.0], dtype=torch.float32)
        policy = ExecutionPolicy(
            providers=("triton", "torch"),
            fixed_provider="triton",
            evidence_requirement="any",
        )
        explanation = runtime.explain_dispatch(
            "testing.scale_shift@1", value, policy=policy
        )
        self.assertIsNone(explanation.selected_implementation_id)
        self.assertIn("not_requested_provider", explanation.candidates[0].reasons)

    def test_unsupported_dtype_is_structurally_rejected(self) -> None:
        runtime = create_phase0_runtime()
        value = torch.tensor([1.0], dtype=torch.float16)
        explanation = runtime.explain_dispatch(
            "testing.scale_shift@1", value, policy=exact_policy()
        )
        self.assertIsNone(explanation.selected_implementation_id)
        self.assertIn("unsupported_dtype", explanation.candidates[0].reasons)

    def test_plan_is_bound_to_call_metadata(self) -> None:
        runtime = create_phase0_runtime()
        value = torch.ones(3, dtype=torch.float32)
        plan = runtime.plan(
            "testing.scale_shift@1",
            value,
            policy=exact_policy(),
            scale=2.0,
        )
        with self.assertRaises(PlanSignatureMismatchError):
            plan.run(torch.ones(4, dtype=torch.float32), scale=2.0)

    def test_torch_autograd_remains_available(self) -> None:
        runtime = create_phase0_runtime()
        value = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
        output = runtime.execute(
            "testing.scale_shift@1",
            value,
            policy=exact_policy(),
            scale=-3.0,
            bias=2.0,
        )
        output.sum().backward()
        torch.testing.assert_close(value.grad, torch.full_like(value, -3.0))

    def test_runtime_failure_never_retries_another_provider(self) -> None:
        class KernelFailure(RuntimeError):
            pass

        runtime = create_phase0_runtime()
        base = runtime.implementations.get(IMPLEMENTATION_ID)
        self.assertIsNotNone(base)
        assert base is not None
        failing_manifest = replace(
            base.manifest,
            implementation_id="splatforge.torch.testing.scale_shift.failing",
            lowering_target="tests:always_fail",
            static_variant_token="phase0-always-fail",
            priority=100,
            build_fingerprint="phase0-failing-fixture",
        )

        def always_fail(input, *, scale, bias):
            raise KernelFailure("simulated provider failure")

        runtime.register_implementation(failing_manifest, always_fail)
        for implementation_id in (IMPLEMENTATION_ID, failing_manifest.implementation_id):
            runtime.record_capability(
                CapabilityRecord(
                    implementation_id=implementation_id,
                    status="conformant",
                    authority="test_fixture",
                    environment_id="cpu-test",
                    release_tier="tier2",
                )
            )
        policy = ExecutionPolicy(
            providers=("torch",),
            evidence_requirement="conformant",
        )
        with self.assertRaisesRegex(KernelFailure, "simulated provider failure"):
            runtime.execute(
                "testing.scale_shift@1",
                torch.tensor([1.0], dtype=torch.float32),
                policy=policy,
                scale=1.0,
                bias=0.0,
            )


if __name__ == "__main__":
    unittest.main()
