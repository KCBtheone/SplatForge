"""Studio recipes for the standalone PGSR and Unbiased Surfel trainers."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import re
import subprocess
import sys

from ..._paths import output_path
from ..artifacts import RunLock
from ..config import StudioConfig, load_config


@dataclass
class ColmapConfig:
    source: str = ""
    downscale: int = 1
    source_root: str = ""


@dataclass
class NativeRenderConfig:
    backend: str = ""


@dataclass
class NativeTrainConfig:
    output: str = "outputs/run"
    device: str = "cuda"
    steps: int = 30_000
    seed: int = 42
    normal_weight: float = 0.015
    gpu_view_cache: bool = False
    export_on_finish: bool = True


@dataclass
class SurfelTrainConfig(NativeTrainConfig):
    normal_weight: float = 0.05
    memory_fraction: float = 0.75
    opacity_reset_policy: str = "refinement"


@dataclass
class NativeExportConfig:
    kind: str = "method"
    voxel_size: float = 0.002
    truncation: float = 0.008


@dataclass
class NativeConfig:
    method: str
    data: ColmapConfig = field(default_factory=ColmapConfig)
    render: NativeRenderConfig = field(default_factory=NativeRenderConfig)
    train: NativeTrainConfig = field(default_factory=NativeTrainConfig)
    export: NativeExportConfig = field(default_factory=NativeExportConfig)

    to_dict = StudioConfig.to_dict
    save = StudioConfig.save

    def validate(self):
        recipe = {"pgsr": PGSR, "unbiased-surfel": UnbiasedSurfel}[self.method]
        if self.render.backend != recipe.backend:
            raise ValueError(f"{self.method} requires render.backend={recipe.backend}")
        if not isinstance(self.data.source, str) or not self.data.source:
            raise ValueError("data.source must point to a COLMAP scene")
        if type(self.data.downscale) is not int or self.data.downscale < 1:
            raise ValueError("data.downscale must be a positive integer")
        if type(self.train.steps) is not int or self.train.steps < 1:
            raise ValueError("train.steps must be a positive integer")
        if type(self.train.seed) is not int or not 0 <= self.train.seed < 2**32:
            raise ValueError("train.seed must be a uint32")
        if not isinstance(self.train.device, str) or not re.fullmatch(r"cuda(?::[0-9]+)?", self.train.device):
            raise ValueError("train.device must be cuda or cuda:N")
        if not math.isfinite(self.train.normal_weight) or self.train.normal_weight < 0:
            raise ValueError("train.normal_weight must be finite and nonnegative")
        if type(self.train.export_on_finish) is not bool or type(self.train.gpu_view_cache) is not bool:
            raise ValueError("train.export_on_finish and train.gpu_view_cache must be boolean")
        if isinstance(self.train, SurfelTrainConfig):
            if not 0 < self.train.memory_fraction <= 1:
                raise ValueError("train.memory_fraction must be in (0, 1]")
            if self.train.opacity_reset_policy not in ("legacy", "refinement"):
                raise ValueError("train.opacity_reset_policy must be legacy or refinement")
        if self.export.kind not in ("method", "mesh", "tsdf"):
            raise ValueError(f"{self.method} supports TSDF mesh export")
        if not math.isfinite(self.export.truncation) or not (
            0 < self.export.voxel_size <= self.export.truncation
        ):
            raise ValueError("TSDF lengths must be finite and truncation must be at least one voxel")
        return self


class NativeMethod:
    exporter = "tsdf"
    requires_prior = False
    train_config = NativeTrainConfig

    @classmethod
    def default_config(cls):
        return NativeConfig(cls.name, render=NativeRenderConfig(cls.backend), train=cls.train_config())

    @classmethod
    def train_run(cls, config, resume=None):
        root = output_path(config.train.output)
        config.data.source = str(Path(config.data.source).expanduser().resolve(strict=True))
        if config.data.source_root:
            config.data.source_root = str(Path(config.data.source_root).expanduser().resolve(strict=True))
        config.train.output = str(root)
        checkpoint = root / "checkpoint.pt"
        settings = root / "config.yaml"
        command = cls.command(config, root)
        with RunLock(root):
            if resume:
                if resume != "latest" and Path(resume).expanduser().resolve() != checkpoint:
                    raise ValueError(f"Resume {cls.name} from {checkpoint} or pass --resume latest")
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                command.extend(cls.resume_args(checkpoint))
            elif checkpoint.exists():
                raise ValueError("Run already has a checkpoint. Pass --resume latest or choose another output.")
            if (root / "checkpoints").exists():
                raise ValueError(f"{root} contains a Studio pipeline run")
            if settings.exists():
                previous = load_config(settings)
                if previous.method != cls.name or previous.data.source != config.data.source:
                    raise ValueError(f"{root} belongs to a different method or dataset")
                if resume and (
                    previous.data.downscale != config.data.downscale
                    or previous.train.seed != config.train.seed
                    or previous.train.normal_weight != config.train.normal_weight
                    or (cls.name == "pgsr" and previous.train.steps != config.train.steps)
                    or (
                        isinstance(config.train, SurfelTrainConfig)
                        and previous.train.opacity_reset_policy != config.train.opacity_reset_policy
                    )
                ):
                    raise ValueError("Resume requires the original training settings")
            if not config.train.export_on_finish:
                command.append("--no-export")
            temporary = settings.with_suffix(".yaml.tmp")
            config.save(temporary)
            temporary.replace(settings)
            subprocess.run(command, check=True)
            if config.train.export_on_finish:
                print(cls.mesh_result(root, config)["path"])

    @classmethod
    def export_run(cls, config, root, device=None):
        root = output_path(root)
        if device:
            config.train.device = device
            config.validate()
        with RunLock(root):
            checkpoint = root / "checkpoint.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            subprocess.run([*cls.command(config, root), *cls.export_args(checkpoint)], check=True)
            return cls.mesh_result(root, config)

    @classmethod
    def mesh_result(cls, root, config):
        mesh = root / "mesh.ply"
        if not mesh.is_file():
            raise RuntimeError(f"{cls.name} did not write {mesh}")
        return {
            "exporter": "tsdf-open3d", "path": str(mesh),
            "voxel_size": config.export.voxel_size, "truncation": config.export.truncation,
        }

    @classmethod
    def command(cls, config, root):
        source, entry = source_paths(config.data.source_root)
        common = [
            "--output", str(root), "--device", config.train.device,
            "--seed", str(config.train.seed), "--factor", str(config.data.downscale),
            "--tsdf-voxel", str(config.export.voxel_size),
            "--tsdf-trunc", str(config.export.truncation),
        ]
        return [sys.executable, "-B", *cls.arguments(config, source, entry), *common]


def source_paths(configured):
    packaged = Path(__file__).resolve().parents[1] / "native_source"
    checkout = Path(__file__).resolve().parents[4]
    entry = packaged / "train.py"
    if not entry.is_file():
        entry = checkout / "deploy/runpod/pgsr.py"
    configured = configured or os.environ.get("PGSR_SOURCE_ROOT")
    if configured:
        source = Path(configured).expanduser().resolve()
    elif (packaged / "tools/commercial_colmap.py").is_file():
        source = packaged
    else:
        candidates = list((checkout / "self_reports").glob("*/source/tools/train_commercial_methods.py"))
        if len(candidates) != 1:
            raise RuntimeError("Use a prepared SplatForge release or set data.source_root to the native source directory")
        source = candidates[0].parents[1]
    if not (source / "tools/commercial_colmap.py").is_file():
        raise FileNotFoundError(source / "tools/commercial_colmap.py")
    return source, entry


class PGSR(NativeMethod):
    name = "pgsr"
    description = "PGSR planar Gaussians with multiview regularization"
    backend = "gsplat-pgsr"

    @staticmethod
    def arguments(config, source, entry):
        if not entry.is_file():
            raise FileNotFoundError(entry)
        args = [
            str(entry), "--studio", "--source-root", str(source), "--data", config.data.source,
            "--steps", str(config.train.steps), "--normal-weight", str(config.train.normal_weight),
        ]
        return [*args, "--gpu-view-cache"] if config.train.gpu_view_cache else args

    @staticmethod
    def resume_args(checkpoint):
        return []

    @staticmethod
    def export_args(checkpoint):
        return ["--export-only"]


class UnbiasedSurfel(NativeMethod):
    name = "unbiased-surfel"
    description = "Unbiased Surfel with corrected depth and convergence regularization"
    backend = "unbiased-native"
    train_config = SurfelTrainConfig

    @staticmethod
    def arguments(config, source, entry):
        script = source / "tools/train_commercial_methods.py"
        if not script.is_file():
            raise FileNotFoundError(script)
        cuda_source = entry.parent / "csrc"
        if not cuda_source.is_dir():
            cuda_source = Path(__file__).resolve().parents[4] / "commercial_unbiased_gsplat/csrc"
        return [
            str(script), "--studio", "--production", "--mode", "unbiased",
            "--cuda-source", str(cuda_source),
            "--data-dir", config.data.source, "--max-steps", str(config.train.steps),
            "--unbiased-normal-weight", str(config.train.normal_weight),
            "--memory-fraction", str(config.train.memory_fraction),
            "--opacity-reset-policy", config.train.opacity_reset_policy,
            "--view-tensor-cache", "gpu" if config.train.gpu_view_cache else "none",
        ]

    @staticmethod
    def resume_args(checkpoint):
        return ["--resume", str(checkpoint)]

    @staticmethod
    def export_args(checkpoint):
        return ["--export-only", str(checkpoint)]
