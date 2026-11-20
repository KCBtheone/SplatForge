"""Configuration for the native VCR-GauS recipe."""

from dataclasses import dataclass, field
import math

import torch

from .config import DataConfig, DensifyConfig, LossConfig, ModelConfig, StudioConfig


@dataclass
class VCRDataConfig(DataConfig):
    normal_prior_key: str = "arr_0"


@dataclass
class VCRModelConfig(ModelConfig):
    opacity_reset_max: float = 0.01


@dataclass
class VCRLossConfig(LossConfig):
    mono_normal: float = 0.0
    consistent_normal: float = 0.0
    depth_normal: float = 0.0
    normal_curvature: float = 0.0
    distortion: float = 0.0
    l1_scale: float = 0.0
    cos_temperature: float = 0.01
    normal_prior_facing: str = "as-is"
    reference_ssim: bool = True
    mono_normal_start: int = 0
    consistent_normal_start: int = 15000
    depth_normal_start: int = 0
    normal_curvature_start: int = 0
    distortion_start: int = 15000
    prune_percent: float = 0.5
    prune_decay: float = 0.6
    prune_v_pow: float = 0.1
    prune_iterations: tuple[int, ...] = (15000, 25000)


@dataclass
class VCRDensifyConfig(DensifyConfig):
    split_rule: str = "max_axis"
    large_percent_dense: float = 0.0
    large_sample_cams: int = 30
    large_max_memory_gib: float = 22.0


@dataclass
class VCRStudioConfig(StudioConfig):
    method: str = "vcrgaus"
    data: VCRDataConfig = field(default_factory=VCRDataConfig)
    model: VCRModelConfig = field(default_factory=VCRModelConfig)
    loss: VCRLossConfig = field(default_factory=VCRLossConfig)
    densify: VCRDensifyConfig = field(default_factory=VCRDensifyConfig)

    def validate(self):
        super().validate()
        if self.render.backend != "vcr-native":
            raise ValueError("VCR-GauS requires the vcr-native renderer")
        if torch.device(self.train.device).type != "cuda":
            raise ValueError("VCR-GauS requires a CUDA device")
        if not 0 < self.model.opacity_reset_max < 1:
            raise ValueError("model.opacity_reset_max must be between zero and one")
        if self.loss.normal_prior_facing not in ("as-is", "camera"):
            raise ValueError("loss.normal_prior_facing must be as-is or camera")
        if self.densify.split_rule not in ("max_axis", "random"):
            raise ValueError("densify.split_rule must be max_axis or random")
        if self.densify.absolute_gradients:
            raise ValueError("The native VCR rasterizer returns one combined screen gradient")
        if type(self.densify.large_sample_cams) is not int or self.densify.large_sample_cams < 1:
            raise ValueError("densify.large_sample_cams must be a positive integer")
        if not math.isfinite(self.densify.large_max_memory_gib) or self.densify.large_max_memory_gib <= 0:
            raise ValueError("densify.large_max_memory_gib must be finite and positive")
        if not math.isfinite(self.densify.large_percent_dense) or self.densify.large_percent_dense < 0:
            raise ValueError("densify.large_percent_dense must be finite and nonnegative")
        return self
