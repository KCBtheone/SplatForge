"""Optional Apache-2.0 DA3-BASE prior adapter with explicit camera alignment.

The default AmbiSuR paper weight is NOT used: its noncommercial license differs
from DA3-BASE. Pose-conditioned BASE depths are aligned to input COLMAP units.
No held-out image may enter a training-prior inference window.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def training_windows(indices, *, size=12, overlap=4):
    if size < 3 or not 0 <= overlap < size:
        raise ValueError("multi-view windows require size>=3 and 0<=overlap<size")
    if len(indices) < 3:
        raise ValueError(
            "pose-conditioned multi-view priors require >=3 training views"
        )
    rows = []
    for start in range(0, len(indices), size - overlap):
        end = min(start + size, len(indices))
        if end - start < 3:
            start = max(0, end - size)
        row = list(indices[start:end])
        if not rows or row != rows[-1]:
            rows.append(row)
        if end == len(indices):
            break
    return rows


def align_prior_pixels(depth, confidence, processed_K, original_K, *, width, height):
    """Resample by camera rays, accounting for resize AND principal-point crop.

    Coordinates follow the external image API's integer pixel centers. Training
    uses its established +0.5 convention independently; no original K is changed.
    Depth is camera-z and remains in the API's input-pose-aligned units.
    """
    depth = torch.as_tensor(np.asarray(depth, np.float32))
    confidence = torch.as_tensor(np.asarray(confidence, np.float32))
    if depth.ndim != 2 or confidence.shape != depth.shape:
        raise ValueError("DA3 depth/confidence must be matching 2D arrays")
    source_h, source_w = depth.shape
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    pixels = torch.stack((x, y, torch.ones_like(x)), -1)
    original_K = torch.as_tensor(np.asarray(original_K, np.float32))
    processed_K = torch.as_tensor(np.asarray(processed_K, np.float32))
    projected = pixels @ torch.linalg.inv(original_K).T @ processed_K.T
    xy = projected[..., :2] / projected[..., 2:3]
    grid = torch.stack(
        (2 * (xy[..., 0] + 0.5) / source_w - 1, 2 * (xy[..., 1] + 0.5) / source_h - 1),
        -1,
    )[None]
    finite = torch.isfinite(depth) & (depth > 0) & torch.isfinite(confidence)
    values = torch.stack(
        (torch.where(finite, depth, 0), torch.where(finite, confidence.clamp_min(0), 0))
    )[None]
    sampled = F.grid_sample(
        values, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )[0]
    valid = (
        (xy[..., 0] >= 0)
        & (xy[..., 0] <= source_w - 1)
        & (xy[..., 1] >= 0)
        & (xy[..., 1] <= source_h - 1)
    )
    return torch.where(valid, sampled[0], 0).numpy(), torch.where(
        valid, sampled[1], 0
    ).numpy()


class DA3BasePriors:
    def __init__(self, model_directory, *, device="cuda", process_res=504):
        from depth_anything_3.api import DepthAnything3

        self.model = (
            DepthAnything3.from_pretrained(str(model_directory), local_files_only=True)
            .to(device)
            .eval()
        )
        self.process_res = process_res

    @torch.no_grad()
    def predict(self, frames):
        extrinsics = np.stack(
            [frame.camera.world_to_camera.numpy() for frame in frames]
        )
        intrinsics = np.stack([frame.camera.K.numpy() for frame in frames])
        prediction = self.model.inference(
            image=[str(frame.image) for frame in frames],
            extrinsics=extrinsics.copy(),
            intrinsics=intrinsics.copy(),
            align_to_input_ext_scale=True,
            infer_gs=False,
            process_res=self.process_res,
            process_res_method="upper_bound_resize",
            ref_view_strategy="middle",
        )
        if not np.allclose(
            prediction.extrinsics, extrinsics[:, :3], atol=1e-4, rtol=1e-4
        ):
            raise ValueError(
                "DA3 did not return the requested input-aligned camera frame"
            )
        rows = []
        for i, frame in enumerate(frames):
            rows.append(
                align_prior_pixels(
                    prediction.depth[i],
                    prediction.conf[i],
                    prediction.intrinsics[i],
                    intrinsics[i],
                    width=frame.camera.width,
                    height=frame.camera.height,
                )
            )
        return rows
