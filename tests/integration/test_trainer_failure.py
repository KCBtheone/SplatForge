from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from splatforge.studio.config import load_config
from splatforge.studio.demo import make_demo
from splatforge.studio.pipeline import GaussianPipeline
from splatforge.studio.trainer import Trainer


class TrainerFailureTests(unittest.TestCase):
    def test_checkpoint_failure_does_not_hide_original_runtime_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = make_demo(root / "data", width=8, points=8, views=3)
            config = load_config(
                method="gof",
                overrides=[
                    f"data.source={data}",
                    f"train.output={root / 'run'}",
                    "train.device=cpu",
                    "render.backend=torch",
                    "train.steps=1",
                    "data.downscale=1",
                    "data.init_points=8",
                ],
            )
            trainer = Trainer(config)
            with (
                patch.object(
                    GaussianPipeline,
                    "training_loss",
                    side_effect=RuntimeError("original failure"),
                ),
                patch.object(
                    trainer, "checkpoint", side_effect=RuntimeError("save failure")
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "original failure"):
                    trainer.run()
            state = json.loads((root / "run/state.json").read_text())
            self.assertEqual(state["stage"], "FAILED")
            self.assertEqual(state["error"], "RuntimeError: original failure")
            self.assertEqual(
                state["checkpoint_save_error"], "RuntimeError: save failure"
            )
            self.assertIsNone(state["recoverable_checkpoint"])
