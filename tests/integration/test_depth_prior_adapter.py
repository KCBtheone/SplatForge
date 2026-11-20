from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np

from splatforge.integrations.depth_anything3 import DA3BasePriors, align_prior_pixels, training_windows


class DepthPriorAdapterTests(unittest.TestCase):
    def test_base_metadata_keeps_explicit_local_loader(self):
        api = ModuleType("depth_anything_3.api")
        api.DepthAnything3 = MagicMock()
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text(json.dumps(
                {"model_name": "da3-base", "config": {"net": {"name": "vitb"}}}
            ))
            # Loader is mocked: this only tests preflight and routing, not weights.
            Path(directory, "model.safetensors").write_bytes(b"mock weight placeholder")
            with patch.dict("sys.modules", {"depth_anything_3.api": api}):
                DA3BasePriors(directory, device="cpu")
            api.DepthAnything3.from_pretrained.assert_called_once_with(
                str(Path(directory).resolve()), local_files_only=True
            )

    def test_partial_weights_rejected_without_sdk_load(self):
        api = ModuleType("depth_anything_3.api")
        api.DepthAnything3 = MagicMock()
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text(json.dumps(
                {"model_name": "da3-base", "config": {"net": {"name": "vitb"}}}
            ))
            partial = Path(directory, "model.safetensors.part")
            partial.write_bytes(b"partial")
            with patch.dict("sys.modules", {"depth_anything_3.api": api}):
                with self.assertRaisesRegex(FileNotFoundError, "resumable partial"):
                    DA3BasePriors(directory, device="cpu")
            api.DepthAnything3.from_pretrained.assert_not_called()
            self.assertEqual(partial.stat().st_size, 7)

    def test_non_base_variant_rejected_before_sdk_import_or_gpu_load(self):
        for metadata in (
            {"model_name": "da3-large", "config": {"net": {"name": "vitl"}}},
            {"model_name": "da3-base", "config": {"net": {"name": "vitg"}}},
            {},
        ):
            with self.subTest(metadata=metadata), tempfile.TemporaryDirectory() as directory:
                Path(directory, "config.json").write_text(json.dumps(metadata))
                with self.assertRaisesRegex(ValueError, "DA3-BASE / vitb"):
                    DA3BasePriors(directory, device="cpu")

    def test_windows_cover_train_frames_without_using_holdouts(self):
        train = [i for i in range(120) if i % 8 != 0]
        windows = training_windows(train)
        self.assertEqual(set().union(*map(set, windows)), set(train))
        self.assertTrue(all(3 <= len(row) <= 12 for row in windows))
        self.assertTrue(all(i % 8 != 0 for row in windows for i in row))

    def test_pixel_alignment_accounts_for_crop_and_keeps_camera_z_units(self):
        original_K = np.array([[2, 0, 2], [0, 2, 1], [0, 0, 1]], np.float32)
        processed_K = original_K.copy()
        processed_K[0, 2] -= 1  # one original pixel cropped from each side
        depth = np.tile(np.array([2, 3, 4], np.float32), (3, 1))
        result, confidence = align_prior_pixels(
            depth, np.ones_like(depth), processed_K, original_K, width=5, height=3
        )
        np.testing.assert_allclose(result[:, 1:4], depth, atol=1e-6)
        np.testing.assert_allclose(result[:, [0, 4]], 0, atol=0)
        np.testing.assert_allclose(confidence[:, 1:4], 1, atol=1e-6)

    def test_bad_window_and_array_shapes_are_rejected(self):
        with self.assertRaises(ValueError):
            training_windows([1, 2])
        with self.assertRaises(ValueError):
            align_prior_pixels(
                np.ones((2, 2)),
                np.ones((3, 3)),
                np.eye(3),
                np.eye(3),
                width=2,
                height=2,
            )
