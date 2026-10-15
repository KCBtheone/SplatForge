from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from splatforge.core.digests import hash_path
from tools.quality.scope_gate import (
    normalize_repo_path,
    path_matches,
    validate_changed_paths,
    validate_frozen_inputs,
)


class ScopeGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.work_order = {
            "allowed_paths": [
                "src/splatforge/providers/triton/matching/**",
                "self_reports/TASK/**",
            ],
            "forbidden_paths": [
                "specs/**",
                "tests/conformance/**",
            ],
            "frozen_inputs": {},
        }

    def test_path_normalization_rejects_traversal_and_absolute_paths(self) -> None:
        for path in ("../specs/x", "/specs/x", "C:/specs/x"):
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    normalize_repo_path(path)

    def test_recursive_allowlist_matches_descendants(self) -> None:
        self.assertTrue(
            path_matches(
                "src/splatforge/providers/triton/matching/kernel.py",
                "src/splatforge/providers/triton/matching/**",
            )
        )

    def test_forbidden_and_unlisted_paths_fail(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            violations = validate_changed_paths(
                [
                    "specs/matching/patch_zncc/opspec.yaml",
                    "README.md",
                    "src/splatforge/providers/triton/matching/kernel.py",
                ],
                self.work_order,
                raw,
            )
        self.assertEqual(
            [item.code for item in violations],
            ["forbidden_path", "outside_allowlist"],
        )

    def test_global_protected_policy_overrides_work_order_allowlist(self) -> None:
        work_order = dict(self.work_order)
        work_order["allowed_paths"] = [
            *self.work_order["allowed_paths"],
            "specs/**",
        ]
        with tempfile.TemporaryDirectory() as raw:
            violations = validate_changed_paths(
                ["specs/op/opspec.yaml"],
                work_order,
                raw,
                protected_patterns=("specs/**",),
            )
        self.assertEqual([item.code for item in violations], ["protected_path"])

    def test_frozen_inputs_are_rehashed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            target = root / "specs" / "op"
            target.mkdir(parents=True)
            (target / "value.txt").write_text("frozen", encoding="utf-8")
            work_order = dict(self.work_order)
            work_order["frozen_inputs"] = {
                "spec": {"path": "specs/op", "digest": hash_path(target)}
            }
            self.assertEqual(validate_frozen_inputs(work_order, root), [])
            (target / "value.txt").write_text("changed", encoding="utf-8")
            violations = validate_frozen_inputs(work_order, root)
            self.assertEqual([item.code for item in violations], ["frozen_input_digest_mismatch"])


if __name__ == "__main__":
    unittest.main()
