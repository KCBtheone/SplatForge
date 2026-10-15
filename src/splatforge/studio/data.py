"""Native COLMAP/JSON parsers and a bounded, lazy image/prior data manager."""

from __future__ import annotations

import json
import struct
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..providers.torch.gaussian.reference import rotation_matrix
from .cameras import Camera
from .config import DataConfig
from .plugins import ComponentRegistry


@dataclass
class Frame:
    camera: Camera
    image: Path
    mask: Path | None = None
    prior: Path | None = None
    split: str | None = None


@dataclass
class ParsedScene:
    frames: list[Frame]
    points: np.ndarray
    colors: np.ndarray


def _existing(path):
    return path if path.is_file() else None


def _read_exact(handle, n):
    value = handle.read(n)
    if len(value) != n:
        raise ValueError("truncated COLMAP binary")
    return value


def _unpack(handle, fmt):
    return struct.unpack("<" + fmt, _read_exact(handle, struct.calcsize("<" + fmt)))


def _intrinsics(model, parameters):
    if model in (0, "SIMPLE_PINHOLE"):
        f, cx, cy = parameters
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], np.float32)
    if model in (1, "PINHOLE"):
        fx, fy, cx, cy = parameters
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32)
    raise ValueError(
        f"COLMAP camera {model} requires undistortion to PINHOLE; distortion is not silently ignored"
    )


def _pose(qvec, translation):
    rotation = rotation_matrix(torch.tensor(qvec, dtype=torch.float64)).numpy()
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3], matrix[:3, 3] = rotation, translation
    return matrix


def parse_colmap(root: Path, config: DataConfig) -> ParsedScene:
    sparse = root / "sparse" / "0"
    if not sparse.is_dir():
        sparse = root / "sparse"
    cameras, images, points, colors = {}, [], [], []
    if (sparse / "cameras.bin").is_file():
        with (sparse / "cameras.bin").open("rb") as handle:
            for _ in range(_unpack(handle, "Q")[0]):
                cid, model, width, height = _unpack(handle, "iiQQ")
                count = {0: 3, 1: 4}.get(model)
                if count is None:
                    raise ValueError(
                        f"camera model {model} must be undistorted before training"
                    )
                K = _intrinsics(model, _unpack(handle, "d" * count))
                cameras[cid] = (width, height, K)
        with (sparse / "images.bin").open("rb") as handle:
            for _ in range(_unpack(handle, "Q")[0]):
                values = _unpack(handle, "idddddddi")
                name = bytearray()
                while (char := _read_exact(handle, 1)) != b"\0":
                    name.extend(char)
                images.append(
                    (
                        values[0],
                        name.decode("utf-8"),
                        values[-1],
                        _pose(values[1:5], values[5:8]),
                    )
                )
                count = _unpack(handle, "Q")[0]
                _read_exact(handle, count * 24)
        with (sparse / "points3D.bin").open("rb") as handle:
            for _ in range(_unpack(handle, "Q")[0]):
                values = _unpack(handle, "QdddBBBd")
                points.append(values[1:4])
                colors.append(values[4:7])
                _read_exact(handle, _unpack(handle, "Q")[0] * 8)
    else:
        for line in (sparse / "cameras.txt").read_text().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            p = line.split()
            cameras[int(p[0])] = (
                int(p[2]),
                int(p[3]),
                _intrinsics(p[1], list(map(float, p[4:]))),
            )
        lines = [
            line
            for line in (sparse / "images.txt").read_text().splitlines()
            if not line.startswith("#")
        ]
        for index in range(0, len(lines), 2):
            if not lines[index].strip():
                continue
            p = lines[index].split()
            images.append(
                (
                    int(p[0]),
                    " ".join(p[9:]),
                    int(p[8]),
                    _pose(list(map(float, p[1:5])), list(map(float, p[5:8]))),
                )
            )
        for line in (sparse / "points3D.txt").read_text().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            p = line.split()
            points.append(list(map(float, p[1:4])))
            colors.append(list(map(float, p[4:7])))
    frames = []
    for _, name, cid, matrix in sorted(images):
        width, height, K = cameras[cid]
        image = root / config.images / name
        frames.append(
            Frame(
                Camera(torch.tensor(K), torch.tensor(matrix), width, height, name),
                image,
                _existing(root / config.masks / Path(name).with_suffix(".png")),
                _existing(root / config.priors / Path(name).with_suffix(".npz")),
            )
        )
    return ParsedScene(
        frames,
        np.asarray(points, np.float32).reshape(-1, 3),
        np.asarray(colors, np.float32).reshape(-1, 3) / 255,
    )


def parse_transforms(root: Path, config: DataConfig) -> ParsedScene:
    path = (
        root / "scene.json"
        if (root / "scene.json").is_file()
        else root / "transforms.json"
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    frames = []
    for row in data["frames"]:
        image = root / row["file_path"]
        if not image.suffix:
            image = image.with_suffix(".png")
        with Image.open(image) as pil:
            actual_width, actual_height = pil.size
        width = int(row.get("w", data.get("w", actual_width)))
        height = int(row.get("h", data.get("h", actual_height)))
        if (width, height) != (actual_width, actual_height):
            raise ValueError(f"camera/image resolution mismatch: {image}")
        fx = float(row.get("fl_x", data.get("fl_x", 0)))
        fy = float(row.get("fl_y", data.get("fl_y", fx)))
        if fx <= 0 and "camera_angle_x" in data:
            fx = 0.5 * width / np.tan(0.5 * float(data["camera_angle_x"]))
            fy = fx
        if fx <= 0 or fy <= 0:
            raise ValueError("transforms require valid focal lengths")
        cx = float(row.get("cx", data.get("cx", width / 2)))
        cy = float(row.get("cy", data.get("cy", height / 2)))
        K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)
        if "world_to_camera" in row:
            matrix = torch.tensor(row["world_to_camera"], dtype=torch.float32)
        else:
            c2w = torch.tensor(row["transform_matrix"], dtype=torch.float32)
            convention = row.get(
                "camera_convention", data.get("camera_convention", "opengl")
            )
            if convention == "opengl":
                c2w = c2w @ torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0]))
            elif convention != "opencv":
                raise ValueError(f"unknown camera convention: {convention}")
            matrix = torch.linalg.inv(c2w)
        relative = Path(row["file_path"])
        prior = (
            root / row["prior_path"]
            if "prior_path" in row
            else root / config.priors / relative.stem
        )
        prior = prior if prior.suffix else prior.with_suffix(".npz")
        mask = (
            root / row["mask_path"]
            if "mask_path" in row
            else root / config.masks / relative.with_suffix(".png").name
        )
        frames.append(
            Frame(
                Camera(K, matrix, width, height, image.name),
                image,
                _existing(mask),
                _existing(prior),
                row.get("split"),
            )
        )
    if "points" in data:
        points = np.asarray(data["points"], np.float32)
        colors = np.asarray(data.get("colors", np.full_like(points, 0.5)), np.float32)
    elif "point_cloud" in data:
        with np.load(root / data["point_cloud"], allow_pickle=False) as cloud:
            points = np.asarray(cloud["points"], np.float32)
            colors = np.asarray(cloud["colors"], np.float32)
    elif (root / "sparse").is_dir():
        cloud = parse_colmap(root, config)
        points, colors = cloud.points, cloud.colors
    else:
        raise ValueError(
            "JSON scenes need points/colors or a point_cloud NPZ with points/colors"
        )
    return ParsedScene(frames, points.reshape(-1, 3), colors.reshape(-1, 3))


parsers = ComponentRegistry("splatforge.dataparsers")
parsers.register("colmap", parse_colmap)
parsers.register("transforms", parse_transforms)


class DataManager:
    def __init__(self, config: DataConfig, background=(1.0, 1.0, 1.0)):
        self.config = config
        self.background = np.asarray(background, np.float32)
        root = Path(config.source).resolve()
        parser = config.parser
        if parser == "auto":
            parser = (
                "transforms"
                if (root / "transforms.json").is_file()
                or (root / "scene.json").is_file()
                else "colmap"
            )
        scene = parsers.get(parser)(root, config)
        if (
            not scene.frames
            or not len(scene.points)
            or not np.isfinite(scene.points).all()
        ):
            raise ValueError(
                "scene must contain finite initialization points and >=1 camera"
            )
        self.frames = scene.frames
        self.points = scene.points
        self.colors = scene.colors
        if (
            self.colors.shape != self.points.shape
            or not np.isfinite(self.colors).all()
            or (self.colors < 0).any()
            or (self.colors > 1).any()
        ):
            raise ValueError(
                "initial point colors must be finite [0,1] RGB values matching the point count"
            )
        self.cache = OrderedDict()
        explicit = any(frame.split is not None for frame in self.frames)
        self.train_indices = (
            [i for i, f in enumerate(self.frames) if (f.split or "train") == "train"]
            if explicit
            else [
                i
                for i in range(len(self.frames))
                if config.test_every == 0 or i % config.test_every != 0
            ]
        )
        self.test_indices = [
            i for i in range(len(self.frames)) if i not in self.train_indices
        ]
        if not self.train_indices:
            self.train_indices = list(range(len(self.frames)))
            self.test_indices = []

    def sample(self, index: int, device="cpu"):
        if index not in self.cache:
            self.cache[index] = self._load(index)
            while len(self.cache) > self.config.cache_images:
                self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        item = self.cache[index]
        return {
            key: value.to(device)
            if isinstance(value, torch.Tensor)
            else value.to(device)
            if isinstance(value, Camera)
            else value
            for key, value in item.items()
        }

    def _load(self, index):
        frame = self.frames[index]
        camera = frame.camera
        width = max(1, camera.width // self.config.downscale)
        height = max(1, camera.height // self.config.downscale)
        with Image.open(frame.image) as pil:
            if pil.size != (camera.width, camera.height):
                raise ValueError(f"camera/image resolution mismatch: {frame.image}")
            rgba = (
                np.asarray(
                    pil.convert("RGBA").resize(
                        (width, height), Image.Resampling.LANCZOS
                    ),
                    np.float32,
                )
                / 255
            )
        mask = rgba[..., 3:4]
        image = rgba[..., :3] * mask + self.background * (1 - mask)
        if frame.mask:
            with Image.open(frame.mask) as pil:
                mask *= (
                    np.asarray(
                        pil.convert("L").resize(
                            (width, height), Image.Resampling.NEAREST
                        ),
                        np.float32,
                    )[..., None]
                    / 255
                )
            image = rgba[..., :3] * mask + self.background * (1 - mask)
        K = camera.K.clone()
        K[0] *= width / camera.width
        K[1] *= height / camera.height
        result = {
            "camera": Camera(K, camera.world_to_camera, width, height, camera.name),
            "image": torch.from_numpy(image),
            "mask": torch.from_numpy(mask),
            "index": index,
        }
        if frame.prior:
            with np.load(frame.prior, allow_pickle=False) as prior:
                for key in ("depth", "normal", "confidence"):
                    if key not in prior:
                        continue
                    value = torch.from_numpy(np.array(prior[key], dtype=np.float32))
                    value = torch.where(torch.isfinite(value), value, 0)
                    if key in ("depth", "confidence"):
                        value = value.clamp_min(0)
                    if value.ndim == 2:
                        value = value.unsqueeze(-1)
                    if value.shape[:2] != (camera.height, camera.width):
                        raise ValueError(
                            f"prior resolution must match original image: {frame.prior}"
                        )
                    value = (
                        torch.nn.functional.interpolate(
                            value.permute(2, 0, 1).unsqueeze(0),
                            size=(height, width),
                            mode="bilinear",
                            align_corners=False,
                        )
                        .squeeze(0)
                        .permute(1, 2, 0)
                    )
                    result["prior_" + key] = value
        return result

    def initialization(self, device, seed=42):
        points, colors = self.points, self.colors
        self.initialization_report = {
            "policy": "unfiltered COLMAP seeds",
            "raw_points": len(points),
            "coordinate_frame": "unchanged input COLMAP",
        }
        if self.config.init_mask_filter == "training-masks":
            from .initialization import mask_consistent_points

            points, colors, self.initialization_report = mask_consistent_points(
                points,
                colors,
                self.frames,
                self.train_indices,
                minimum_views=self.config.init_min_mask_views,
            )
        if self.config.init_from_priors:
            accumulated = []
            rgb = []
            from ..providers.torch.gaussian.reference import pixel_rays

            for index in self.train_indices:
                sample = self.sample(index)
                if "prior_depth" not in sample:
                    raise ValueError(
                        "init_from_priors requires aligned metric depth for every training frame"
                    )
                camera = sample["camera"]
                depth = sample["prior_depth"].squeeze(-1)
                valid = (
                    (depth > 0)
                    & torch.isfinite(depth)
                    & (sample["mask"].squeeze(-1) > 0)
                )
                if "prior_confidence" in sample:
                    confidence = sample["prior_confidence"].squeeze(-1)
                    valid &= confidence >= torch.quantile(confidence[valid], 0.8)
                local = pixel_rays(
                    camera.K, width=camera.width, height=camera.height
                ) * depth.unsqueeze(-1)
                world = local @ camera.camera_to_world[:3, :3].T + camera.origin
                accumulated.append(world[valid].numpy())
                rgb.append(sample["image"][valid].numpy())
            points = np.concatenate(accumulated)
            colors = np.concatenate(rgb)
            self.initialization_report = {
                "policy": "aligned metric priors with training masks and explicit confidence quantile",
                "prior_points_before_sampling": len(points),
                "training_views": len(self.train_indices),
                "heldout_images_or_masks_used": False,
                "coordinate_frame": "unchanged input COLMAP",
            }
        if len(points) > self.config.init_points:
            index = np.random.default_rng(seed).choice(
                len(points), self.config.init_points, replace=False
            )
            points, colors = points[index], colors[index]
        return torch.tensor(points, device=device), torch.tensor(colors, device=device)
