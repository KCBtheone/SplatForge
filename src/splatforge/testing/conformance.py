"""Small, provider-fixed conformance harness used by Phase 0 and later suites."""

from __future__ import annotations

import json
import platform
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from ..core.digests import canonical_json_digest
from ..core.errors import SplatForgeError
from ..core.policy import ExecutionPolicy
from ..core.runtime import Runtime


@dataclass(frozen=True, slots=True)
class ConformanceCase:
    name: str
    operands: tuple[Any, ...]
    attributes: Mapping[str, Any]
    expected: Any
    comparator: Callable[[Any, Any], None] | None = field(default=None, compare=False)


def _assert_equal(actual: Any, expected: Any) -> None:
    if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        return
    if isinstance(actual, (tuple, list)) and isinstance(expected, (tuple, list)):
        if len(actual) != len(expected):
            raise AssertionError("output sequence lengths differ")
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_equal(actual_item, expected_item)
        return
    if actual != expected:
        raise AssertionError(f"{actual!r} != {expected!r}")


def _error_record(exc: BaseException) -> dict[str, Any]:
    if isinstance(exc, SplatForgeError):
        return exc.to_dict()
    return {
        "code": "unstructured_exception",
        "message": str(exc),
        "details": {"type": type(exc).__name__},
    }


def _environment() -> dict[str, Any]:
    raw = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
    }
    raw["environment_id"] = canonical_json_digest(raw)
    return raw


def run_conformance(
    runtime: Runtime,
    op: str,
    cases: Sequence[ConformanceCase],
    *,
    policy: ExecutionPolicy,
    repository_revision: str,
    artifact_digest: str,
    conformance_digest: str,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    if not cases:
        raise ValueError("conformance suite must contain at least one case")
    results: list[dict[str, Any]] = []
    subject: dict[str, Any] | None = None

    for case in cases:
        started = time.perf_counter_ns()
        status = "passed"
        error: dict[str, Any] | None = None
        try:
            plan = runtime.plan(
                op,
                *case.operands,
                policy=policy,
                **dict(case.attributes),
            )
            actual = plan.run(*case.operands, **dict(case.attributes))
            (case.comparator or _assert_equal)(actual, case.expected)
            if subject is None:
                manifest = plan.implementation.manifest
                subject = {
                    "repository_revision": repository_revision,
                    "artifact_digest": artifact_digest,
                    "op": str(manifest.op_key),
                    "spec_revision": manifest.op_spec_revision,
                    "conformance_digest": conformance_digest,
                    "implementation_id": manifest.implementation_id,
                    "build_fingerprint": manifest.build_fingerprint,
                    "snapshot_token": plan.snapshot_token,
                }
        except Exception as exc:  # Preserve every test failure in raw evidence.
            status = "failed"
            error = _error_record(exc)
        duration_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        result: dict[str, Any] = {
            "name": case.name,
            "status": status,
            "duration_ms": duration_ms,
        }
        if error is not None:
            result["error"] = error
        results.append(result)

    passed = sum(item["status"] == "passed" for item in results)
    failed = sum(item["status"] == "failed" for item in results)
    blocked = sum(item["status"] == "blocked" for item in results)
    if subject is None:
        explanation = runtime.explain_dispatch(
            op,
            *cases[0].operands,
            policy=policy,
            **dict(cases[0].attributes),
        )
        selected = explanation.selected_implementation_id or "unresolved"
        subject = {
            "repository_revision": repository_revision,
            "artifact_digest": artifact_digest,
            "op": explanation.op_key,
            "spec_revision": explanation.spec_revision,
            "conformance_digest": conformance_digest,
            "implementation_id": selected,
            "build_fingerprint": "unresolved",
            "snapshot_token": explanation.snapshot_token,
        }

    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    report_seed = {
        "generated_at": generated_at,
        "subject": subject,
        "summary": {"passed": passed, "failed": failed, "blocked": blocked},
    }
    report = {
        "schema_version": 1,
        "report_id": canonical_json_digest(report_seed),
        "report_type": "conformance",
        "authority": "self_reported",
        "generated_at": generated_at,
        "status": "passed" if failed == 0 and blocked == 0 else "failed",
        "subject": subject,
        "environment": _environment(),
        "summary": {
            "passed": passed,
            "failed": failed,
            "blocked": blocked,
            "total": len(results),
        },
        "cases": results,
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report
