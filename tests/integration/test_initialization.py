from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from splatforge.studio.cameras import Camera
from splatforge.studio.initialization import mask_consistent_points


def test_only_training_silhouettes_select_seeds_without_coordinate_changes():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:6, 2:6] = 255
        Image.fromarray(mask).save(root / "train.png")
        Image.fromarray(np.zeros_like(mask)).save(root / "heldout.png")
        camera = Camera(
            torch.tensor([[4.0, 0, 4.0], [0, 4.0, 4.0], [0, 0, 1.0]]),
            torch.eye(4),
            8,
            8,
        )
        frames = [
            SimpleNamespace(camera=camera, mask=root / name)
            for name in ("train.png", "train.png", "heldout.png")
        ]
        points = np.array([[0, 0, 2.0], [1.5, 0, 2.0], [0, 0, -2.0]], dtype=np.float32)
        colors = np.ones_like(points) * 0.5
        selected, rgb, report = mask_consistent_points(points, colors, frames, [0, 1])
        np.testing.assert_array_equal(selected, points[:1])
        np.testing.assert_array_equal(rgb, colors[:1])
        assert report["raw_points"] == 3
        assert report["selected_points"] == 1
        assert report["heldout_images_or_masks_used"] is False
        assert len(points) == 3
