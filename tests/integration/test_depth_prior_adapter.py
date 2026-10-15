from __future__ import annotations

import unittest

import numpy as np

from splatforge.integrations.depth_anything3 import align_prior_pixels, training_windows


class DepthPriorAdapterTests(unittest.TestCase):
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
