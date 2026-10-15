"""Rendering metrics and protocol-explicit point-cloud geometry metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree

from .artifacts import atomic_json
from .losses import ssim_loss


def save_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = (image.detach().cpu().clamp(0, 1).numpy() * 255).round().astype(np.uint8)
    if array.shape[-1] == 1:
        array = array[..., 0]
    Image.fromarray(array).save(path)


@torch.no_grad()
def evaluate_views(pipeline, output, *, save_renders=True, split="test"):
    indices = (
        pipeline.data.test_indices if split == "test" else pipeline.data.train_indices
    )
    if not indices:
        return {
            "split": split,
            "views": 0,
            "metrics": None,
            "reason": "no held-out views",
        }
    rows = []
    for index in indices:
        sample = pipeline.data.sample(index, pipeline.model.device)
        render = pipeline.render(sample["camera"])
        error = (render.rgb - sample["image"]).square().mean().clamp_min(1e-12)
        rows.append(
            {
                "index": index,
                "name": sample["camera"].name,
                "psnr": float(-10 * error.log10()),
                "ssim": float(1 - ssim_loss(render.rgb, sample["image"])),
            }
        )
        if save_renders:
            root = Path(output) / "renders" / split
            stem = f"{index:05d}"
            save_image(root / (stem + "_rgb.png"), render.rgb)
            save_image(root / (stem + "_normal.png"), render.normals * 0.5 + 0.5)
            root.mkdir(parents=True, exist_ok=True)
            np.save(
                root / (stem + "_depth.npy"),
                render.depth.cpu().numpy(),
                allow_pickle=False,
            )
    result = {
        "split": split,
        "views": len(rows),
        "metrics": {
            key: sum(row[key] for row in rows) / len(rows) for key in ("psnr", "ssim")
        },
        "per_view": rows,
    }
    atomic_json(Path(output) / ("evaluation_" + split + ".json"), result)
    return result


def geometry_metrics(
    predicted, ground_truth, *, threshold=0.005, unit="world", transform=None
):
    predicted = np.asarray(predicted, np.float64)
    ground_truth = np.asarray(ground_truth, np.float64)
    if (
        predicted.ndim != 2
        or ground_truth.ndim != 2
        or predicted.shape[1] != 3
        or ground_truth.shape[1] != 3
        or not len(predicted)
        or not len(ground_truth)
    ):
        raise ValueError("geometry metrics require two nonempty Nx3 point sets")
    if (
        not np.isfinite(predicted).all()
        or not np.isfinite(ground_truth).all()
        or threshold <= 0
    ):
        raise ValueError("geometry and threshold must be finite/valid")
    if transform is not None:
        transform = np.asarray(transform, np.float64)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("alignment must be finite 4x4")
        predicted = predicted @ transform[:3, :3].T + transform[:3, 3]
    accuracy = cKDTree(ground_truth).query(predicted, workers=-1)[0]
    completeness = cKDTree(predicted).query(ground_truth, workers=-1)[0]
    precision = float(np.mean(accuracy <= threshold))
    recall = float(np.mean(completeness <= threshold))
    return {
        "protocol": "bidirectional-nearest-neighbor; no implicit ICP, crop or benchmark mask",
        "unit": unit,
        "threshold": threshold,
        "prediction_count": len(predicted),
        "ground_truth_count": len(ground_truth),
        "accuracy": float(accuracy.mean()),
        "completeness": float(completeness.mean()),
        "chamfer_l1": float((accuracy.mean() + completeness.mean()) / 2),
        "precision": precision,
        "recall": recall,
        "f_score": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
    }
