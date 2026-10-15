"""Strict, serializable configurations; algorithm choices are explicit."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    source: str = ""
    parser: str = "auto"
    downscale: int = 2
    images: str = "images"
    masks: str = "masks"
    priors: str = "priors"
    prior_kind: str = "metric"
    test_every: int = 8
    cache_images: int = 8
    init_points: int = 100_000
    init_from_priors: bool = False
    init_mask_filter: str = "none"
    init_min_mask_views: int = 2


@dataclass
class RenderConfig:
    backend: str = "torch"
    tile_size: int = 16
    sigma_extent: float = 3.0
    eps2d: float = 0.3
    near: float = 0.01
    far: float = 10000.0
    background: tuple[float, float, float] = (1.0, 1.0, 1.0)
    antialiased: bool = False
    tile_batch_size: int = 64
    tile_event_budget: int = 1_048_576


@dataclass
class ModelConfig:
    sh_degree: int = 3
    sh_degree_interval: int = 1000
    opacity: float = 0.1
    initial_scale: float = 1.0
    appearance: bool = False


@dataclass
class LossConfig:
    ssim: float = 0.2
    depth_distortion: float = 100.0
    normal: float = 0.05
    geometric_start: int = 1000
    prior_depth: float = 0.1
    ambiguity_normal: float = 0.1
    ambiguity_start: int = 7000
    ambiguity_upper: float = 0.05
    ambiguity_lower: float = 0.10
    trunc_sigma: float = 2.0
    ray_color: float = 1e-5
    ray_color_stop: int = 15000
    multiview: float = 0.0
    multiview_geometry: float = 0.03
    multiview_photometric: float = 0.15
    multiview_pixel_threshold: float = 1.0
    patch_radius: int = 3
    patch_stride: int = 4


@dataclass
class DensifyConfig:
    enabled: bool = True
    start: int = 500
    stop: int = 15000
    interval: int = 100
    gradient_threshold: float = 0.0002
    prune_opacity: float = 0.005
    max_gaussians: int = 2_000_000
    reset_opacity_every: int = 3000
    absolute_gradients: bool = False
    gradient_space: str = "ndc"
    grow_scale_fraction: float = 0.01
    prune_scale_fraction: float = 0.1


@dataclass
class TrainConfig:
    output: str = "outputs/run"
    device: str = "cuda"
    steps: int = 30000
    seed: int = 42
    means_lr: float = 0.00016
    means_lr_final: float = 0.0000016
    spatial_lr_scale: str = "camera-radius"
    features_lr: float = 0.0025
    opacity_lr: float = 0.05
    scales_lr: float = 0.005
    quats_lr: float = 0.001
    appearance_lr: float = 0.001
    checkpoint_every: int = 1000
    eval_every: int = 1000
    log_every: int = 10
    retain_checkpoints: int = 2
    export_on_finish: bool = True


@dataclass
class ExportConfig:
    kind: str = "method"
    voxel_size: float = 0.004
    truncation: float = 0.016
    depth_max: float = 10000.0
    alpha_min: float = 0.5
    level: float = 0.5
    binary_steps: int = 8
    query_chunk: int = 1024
    gaussian_chunk: int = 512
    max_gaussians: int | None = None
    seed_opacity_min: float = 0.05
    tetra_extent: float = 3.0
    tsdf_backend: str = "adaptive"
    min_component_faces: int = 0
    mask_background: bool = False


@dataclass
class StudioConfig:
    method: str = "radegs"
    data: DataConfig = field(default_factory=DataConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    densify: DensifyConfig = field(default_factory=DensifyConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    def validate(self):
        if not self.data.source:
            raise ValueError("data.source is required")
        for name, value in (
            ("data.downscale", self.data.downscale),
            ("train.steps", self.train.steps),
            ("data.cache_images", self.data.cache_images),
            ("data.init_points", self.data.init_points),
            ("data.init_min_mask_views", self.data.init_min_mask_views),
            ("render.tile_size", self.render.tile_size),
            ("render.tile_batch_size", self.render.tile_batch_size),
            ("render.tile_event_budget", self.render.tile_event_budget),
            ("densify.max_gaussians", self.densify.max_gaussians),
            ("export.query_chunk", self.export.query_chunk),
            ("export.gaussian_chunk", self.export.gaussian_chunk),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.data.test_every < 0 or self.train.retain_checkpoints < 1:
            raise ValueError(
                "test_every must be >=0 and at least one checkpoint retained"
            )
        if self.train.log_every < 1 or self.densify.interval < 1:
            raise ValueError("log_every and densify.interval must be positive")
        if self.densify.start < 0 or self.densify.stop < self.densify.start:
            raise ValueError("densification start/stop are invalid")
        if self.densify.gradient_space not in ("ndc", "pixel"):
            raise ValueError("densification gradient_space must be ndc or pixel")
        if self.train.spatial_lr_scale not in ("camera-radius", "unit"):
            raise ValueError("spatial_lr_scale must be camera-radius or unit")
        if not 0 < self.densify.grow_scale_fraction < self.densify.prune_scale_fraction:
            raise ValueError(
                "densification scale fractions must be positive and ordered"
            )
        if (
            not 0 <= self.loss.ssim <= 1
            or self.loss.patch_stride < 1
            or self.loss.patch_radius < 0
        ):
            raise ValueError("SSIM weight and patch sampling settings are invalid")
        if not 0 <= self.model.sh_degree <= 3 or not 0 < self.model.opacity < 1:
            raise ValueError("SH degree must be 0..3; opacity must be in (0,1)")
        if not (
            0 <= self.loss.ambiguity_lower < 1 and 0 <= self.loss.ambiguity_upper < 1
        ):
            raise ValueError("ambiguity quantiles must be in [0,1)")
        if not 0 < self.render.near < self.render.far:
            raise ValueError("render near/far bounds invalid")
        if (
            self.loss.trunc_sigma <= 0
            or self.export.truncation <= 0
            or self.export.voxel_size <= 0
        ):
            raise ValueError("truncation and voxel sizes must be positive")
        if self.data.prior_kind not in ("metric", "mono"):
            raise ValueError("prior_kind must be metric or mono")
        if self.data.init_mask_filter not in ("none", "training-masks"):
            raise ValueError("init_mask_filter must be none or training-masks")
        if self.data.init_from_priors and self.data.init_mask_filter != "none":
            raise ValueError(
                "prior initialization already uses training masks; do not also filter COLMAP seeds"
            )
        if self.export.max_gaussians is not None and self.export.max_gaussians < 4:
            raise ValueError("explicit export.max_gaussians must be >=4")
        if len(self.render.background) != 3:
            raise ValueError("background must have three channels")
        return self

    def to_dict(self):
        return asdict(self)

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8"
        )


def update_dataclass(target, values: dict[str, Any]):
    definitions = {item.name: item for item in fields(target)}
    for name, value in values.items():
        if name not in definitions:
            raise ValueError(
                f"unknown configuration field {type(target).__name__}.{name}"
            )
        current = getattr(target, name)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(f"{name} must be a mapping")
            update_dataclass(current, value)
        else:
            setattr(target, name, tuple(value) if isinstance(current, tuple) else value)
    return target


def load_config(
    path: str | Path | None = None, *, method: str | None = None, overrides=()
) -> StudioConfig:
    from .methods import methods

    content = yaml.safe_load(Path(path).read_text(encoding="utf-8")) if path else {}
    if not isinstance(content or {}, dict):
        raise ValueError("configuration must be a mapping")
    name = method or (content or {}).get("method", "radegs")
    config = methods.get(name).default_config()
    update_dataclass(config, content or {})
    if method:
        config.method = method
    for override in overrides:
        key, separator, raw = override.partition("=")
        if not separator:
            raise ValueError("overrides use dotted.path=value")
        owner = config
        parts = key.split(".")
        for part in parts[:-1]:
            if not hasattr(owner, part) or not is_dataclass(getattr(owner, part)):
                raise ValueError(f"unknown configuration path {key}")
            owner = getattr(owner, part)
        update_dataclass(owner, {parts[-1]: yaml.safe_load(raw)})
    return config.validate()
