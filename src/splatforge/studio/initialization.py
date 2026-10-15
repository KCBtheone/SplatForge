"""Explicit, train-mask-consistent COLMAP seed selection; no coordinate change."""

from __future__ import annotations

import numpy as np
from PIL import Image


def mask_consistent_points(points, colors, frames, indices, *, minimum_views=2):
    foreground = np.zeros(len(points), np.int32)
    background = np.zeros(len(points), np.int32)
    for index in indices:
        frame = frames[index]
        if frame.mask is None:
            raise ValueError(
                "training-masks initialization requires every training mask"
            )
        camera = frame.camera
        pose = camera.world_to_camera.numpy()
        local = points @ pose[:3, :3].T + pose[:3, 3]
        projection = local @ camera.K.numpy().T
        valid_depth = local[:, 2] > 0
        pixels = projection[:, :2] / np.where(valid_depth, projection[:, 2], 1)[:, None]
        visible = (
            valid_depth
            & np.isfinite(pixels).all(-1)
            & (pixels[:, 0] >= 0)
            & (pixels[:, 0] < camera.width)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < camera.height)
        )
        rows = np.flatnonzero(visible)
        with Image.open(frame.mask) as image:
            if image.size != (camera.width, camera.height):
                raise ValueError(
                    "initialization mask must match the original calibrated image"
                )
            mask = np.asarray(image.convert("L"))
        xy = np.floor(pixels[rows]).astype(np.int64)
        inside = mask[xy[:, 1], xy[:, 0]] > 0
        foreground[rows[inside]] += 1
        background[rows[~inside]] += 1
    keep = (foreground >= minimum_views) & (background == 0)
    if not keep.any():
        raise ValueError(
            "no COLMAP point satisfies the explicit training-mask initialization rule"
        )
    return (
        points[keep],
        colors[keep],
        {
            "policy": "training-masks: inside every observed silhouette; minimum foreground views",
            "minimum_views": minimum_views,
            "training_views": len(indices),
            "raw_points": len(points),
            "selected_points": int(keep.sum()),
            "rejected_points": int((~keep).sum()),
            "coordinate_frame": "unchanged input COLMAP",
            "heldout_images_or_masks_used": False,
        },
    )
