from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

import jsonschema

from tools.quality.capture_environment import _gpus, capture


class EnvironmentCaptureTests(unittest.TestCase):
    def test_missing_nvidia_smi_does_not_fail_cpu_environment_capture(self):
        with patch("tools.quality.capture_environment.subprocess.run", side_effect=FileNotFoundError):
            self.assertEqual(_gpus(), [])

    def test_capture_matches_environment_lock_schema(self) -> None:
        root = Path(__file__).resolve().parents[2]
        schema = json.loads(
            (root / "tools/schemas/environment_lock.schema.json").read_text(
                encoding="utf-8"
            )
        )
        jsonschema.Draft202012Validator(
            schema,
            format_checker=jsonschema.FormatChecker(),
        ).validate(capture())


if __name__ == "__main__":
    unittest.main()

