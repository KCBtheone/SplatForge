"""Small, analytically rendered sphere dataset for end-to-end checks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def make_demo(root, *, width=32, height=32, points=96, views=4):
    root = Path(root).resolve()
    if (root / "scene.json").exists():
        raise ValueError("demo target already contains a scene")
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "priors").mkdir(exist_ok=True)
    i = np.arange(points)
    z = 1 - 2 * (i + 0.5) / points
    phi = i * np.pi * (3 - np.sqrt(5))
    normals = np.stack(
        (np.cos(phi) * np.sqrt(1 - z * z), np.sin(phi) * np.sqrt(1 - z * z), z), -1
    )
    center = np.array([0.0, 0.0, 2.0])
    xyz = center + 0.5 * normals
    colors = 0.35 + 0.3 * (normals + 1) / 2
    np.savez(
        root / "points.npz",
        points=xyz.astype(np.float32),
        colors=colors.astype(np.float32),
    )
    x, y = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    focal = width * 1.2
    rays = np.stack(
        ((x - width / 2) / focal, (y - height / 2) / focal, np.ones_like(x)), -1
    )
    frames = []
    for index in range(views):
        angle = (index - (views - 1) / 2) * 0.25
        origin = np.array([0.4 * np.sin(angle), 0.15 * np.cos(angle), 0.0])
        forward = center - origin
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.array([0.0, -1.0, 0.0]))
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        rotation = np.column_stack((right, down, forward))
        directions = rays @ rotation.T
        delta = origin - center
        a = (directions**2).sum(-1)
        b = 2 * (directions * delta).sum(-1)
        c = (delta**2).sum() - 0.25
        discriminant = b * b - 4 * a * c
        valid = discriminant > 0
        depth = np.where(
            valid, (-b - np.sqrt(np.maximum(discriminant, 0))) / (2 * a), 0
        )
        surface = origin + directions * depth[..., None]
        normal = (surface - center) / 0.5
        color = 0.35 + 0.3 * (normal + 1) / 2
        image = np.where(valid[..., None], color, 1.0).clip(0, 1)
        Image.fromarray((image * 255).round().astype(np.uint8)).save(
            root / "images" / f"{index:03d}.png"
        )
        np.savez(
            root / "priors" / f"{index:03d}.npz",
            depth=depth.astype(np.float32),
            normal=(normal @ rotation).astype(np.float32),
            confidence=valid.astype(np.float32),
        )
        c2w = np.eye(4)
        c2w[:3, :3] = rotation
        c2w[:3, 3] = origin
        frames.append(
            {
                "file_path": f"images/{index:03d}.png",
                "prior_path": f"priors/{index:03d}.npz",
                "camera_convention": "opencv",
                "transform_matrix": c2w.tolist(),
                "split": "test" if index == views - 1 else "train",
            }
        )
    content = {
        "w": width,
        "h": height,
        "fl_x": focal,
        "fl_y": focal,
        "cx": width / 2,
        "cy": height / 2,
        "point_cloud": "points.npz",
        "frames": frames,
    }
    (root / "scene.json").write_text(json.dumps(content, indent=2), encoding="utf-8")
    return root
