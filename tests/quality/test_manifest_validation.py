from __future__ import annotations

import unittest
from pathlib import Path

from tools.quality.manifest_validation import (
    _validate_frozen_numeric_artifact,
    find_placeholders,
    validate_repository,
)


class ManifestValidationTests(unittest.TestCase):
    def test_repository_manifests_are_consistent(self) -> None:
        root = Path(__file__).resolve().parents[2]
        issues = validate_repository(root)
        self.assertEqual([issue.to_dict() for issue in issues], [])

    def test_frozen_placeholder_search_is_recursive(self) -> None:
        value = {"outer": [{"digest": "TBD_SHA256"}]}
        self.assertEqual(find_placeholders(value), ["$.outer[0].digest"])

    def test_unmeasured_tolerance_cannot_be_frozen(self) -> None:
        value = {
            "status": "frozen",
            "spec_revision": "sha256:" + "1" * 64,
            "calibration": {
                "status": "unmeasured",
                "rationale": "proposal only",
            },
        }
        issues = _validate_frozen_numeric_artifact(
            value,
            "tests/tolerances/example.yaml",
            kind="tolerance table",
        )
        messages = [issue.message for issue in issues]
        self.assertTrue(any("measured calibration" in message for message in messages))
        self.assertTrue(any("environment_id" in message for message in messages))
        self.assertTrue(any("evidence_digest" in message for message in messages))


if __name__ == "__main__":
    unittest.main()
