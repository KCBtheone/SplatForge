"""Reusable training engine with exact optimizer/RNG/sampler checkpoint state."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .artifacts import (
    RunLock,
    StorageManifest,
    atomic_json,
    load_checkpoint,
    save_checkpoint,
    timestamp,
)
from .evaluation import evaluate_views
from .methods import methods
from .resources import wait_for_capacity


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def random_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_random_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def check_resume_configuration(current, stored):
    # Execution location and completion/observation cadence may change. Method,
    # dataset, representation and losses may not change silently on resume.
    a = current.to_dict()
    b = json.loads(json.dumps(stored))
    a = json.loads(json.dumps(a))
    for record in (a, b):
        for key in (
            "output",
            "device",
            "steps",
            "checkpoint_every",
            "eval_every",
            "log_every",
            "retain_checkpoints",
            "export_on_finish",
        ):
            record["train"].pop(key, None)
        record.pop("export", None)
    if a != b:
        raise ValueError(
            "resume configuration changes method/data/render/loss/optimizer semantics; start a new run"
        )


class Trainer:
    def __init__(self, config, *, resume=None):
        self.config = config.validate()
        self.root = Path(config.train.output).resolve()
        self.resume = resume
        self.pipeline = None
        self.optimizer = None
        self.step = 0
        self.order = []
        self.order_offset = 0
        self.manifest = None

    def write_state(self, stage, **extra):
        atomic_json(
            self.root / "state.json",
            {
                "stage": stage,
                "step": self.step,
                "method": self.config.method,
                "updated": timestamp(),
                **(self.pipeline.statistics() if self.pipeline else {}),
                **extra,
            },
        )

    def setup(self):
        seed_all(self.config.train.seed)
        self.pipeline = methods.get(self.config.method).make_pipeline(self.config)
        if self.resume:
            path = (
                self.root / "checkpoints" / "latest.pt"
                if self.resume == "latest"
                else Path(self.resume)
            )
            checkpoint = load_checkpoint(path, self.config.train.device)
            if checkpoint.get("format") != "splatforge-studio-v1":
                raise ValueError("unsupported checkpoint format")
            check_resume_configuration(self.config, checkpoint["config"])
            self.pipeline.load_state_dict(checkpoint["pipeline"])
            self.step = checkpoint["step"]
            self.order = checkpoint["sampler"]["order"]
            self.order_offset = checkpoint["sampler"]["offset"]
            self.optimizer = torch.optim.Adam(
                self.pipeline.parameter_groups(), eps=1e-15
            )
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            restore_random_state(checkpoint["random"])
        else:
            self.optimizer = torch.optim.Adam(
                self.pipeline.parameter_groups(), eps=1e-15
            )
        self.manifest = StorageManifest(self.root)
        atomic_json(
            self.root / "initialization.json",
            {
                **self.pipeline.data.initialization_report,
                "initialized_gaussians": self.pipeline.model.count,
                "camera_extent": float(self.pipeline.camera_extent),
                "spatial_lr_scale": self.config.train.spatial_lr_scale,
                "coordinate_normalization": "none",
            },
        )
        self.manifest.record(
            self.root / "initialization.json",
            "FINAL",
            "explicit initialization and coordinate policy",
            state="READY",
        )
        self.config.save(self.root / "config.yaml")
        self.manifest.record(
            self.root / "config.yaml",
            "FINAL",
            "method and run configuration",
            state="READY",
        )
        self.write_state("TRAINING")

    def sample_index(self):
        if self.order_offset >= len(self.order):
            self.order = list(self.pipeline.data.train_indices)
            random.shuffle(self.order)
            self.order_offset = 0
        index = self.order[self.order_offset]
        return index

    def checkpoint(self, *, final=False):
        payload = {
            "format": "splatforge-studio-v1",
            "step": self.step,
            "config": self.config.to_dict(),
            "pipeline": self.pipeline.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "random": random_state(),
            "sampler": {"order": self.order, "offset": self.order_offset},
        }
        directory = self.root / "checkpoints"
        path = directory / f"step_{self.step:08d}.pt"
        save_checkpoint(path, payload)
        self.manifest.record(
            path,
            "FINAL" if final else "NEXT",
            "trained scene" if final else "recoverable checkpoint",
            next_use=None if final else "resume training or finish exports",
            release_after=None if final else "a newer checkpoint has been saved",
            state="READY",
        )
        latest = directory / "latest.pt"
        save_checkpoint(latest, payload)
        self.manifest.record(
            latest,
            "FINAL" if final else "NEXT",
            "current restart state",
            next_use=None if final else "resume current run",
            state="READY",
        )
        candidates = sorted(directory.glob("step_*.pt"))
        for old in candidates[: -self.config.train.retain_checkpoints]:
            record = self.manifest.items.get(str(old.resolve()))
            if record and record["decision"] != "FINAL":
                self.manifest.record(
                    old,
                    "RECLAIM",
                    "superseded restart checkpoint",
                    release_after="new checkpoint successfully saved",
                    state="READY",
                )
                self.manifest.reclaim_checkpoint(old)
        return path

    def run(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with RunLock(self.root):
            if (self.root / "checkpoints" / "latest.pt").exists() and not self.resume:
                raise ValueError(
                    "run already has a checkpoint; pass --resume latest or choose a new output"
                )
            wait_for_capacity(self.config, self.write_state)
            self.setup()
            last_disk_check = time.monotonic()
            try:
                log = self.root / "metrics.jsonl"
                with log.open("a", encoding="utf-8") as handle:
                    while self.step < self.config.train.steps:
                        if time.monotonic() - last_disk_check >= 60:
                            self.checkpoint()
                            wait_for_capacity(
                                self.config, self.write_state, check_gpu=False
                            )
                            self.write_state("TRAINING")
                            last_disk_check = time.monotonic()
                        self.pipeline.model.train()
                        self.optimizer.zero_grad(set_to_none=True)
                        self.pipeline.update_learning_rates(self.step, self.optimizer)
                        sample = self.pipeline.data.sample(
                            self.sample_index(), self.config.train.device
                        )
                        loss, terms, render = self.pipeline.training_loss(
                            sample, self.step
                        )
                        if not torch.isfinite(loss):
                            raise FloatingPointError("non-finite training loss")
                        loss.backward()
                        if any(
                            parameter.grad is not None
                            and not torch.isfinite(parameter.grad).all()
                            for parameter in self.pipeline.parameters()
                        ):
                            raise FloatingPointError("non-finite training gradient")
                        self.pipeline.after_backward(render)
                        self.optimizer.step()
                        self.order_offset += 1
                        self.step += 1
                        self.pipeline.after_step(self.step, self.optimizer)
                        if (
                            self.step == 1
                            or self.step % self.config.train.log_every == 0
                            or self.step == self.config.train.steps
                        ):
                            row = {
                                "step": self.step,
                                "loss": float(loss.detach()),
                                "terms": {
                                    key: float(value.detach())
                                    for key, value in terms.items()
                                },
                                **self.pipeline.statistics(),
                                "time": timestamp(),
                            }
                            handle.write(json.dumps(row, allow_nan=False) + "\n")
                            handle.flush()
                            self.write_state("TRAINING", loss=row["loss"])
                            print(json.dumps(row, allow_nan=False), flush=True)
                        if (
                            self.config.train.checkpoint_every > 0
                            and self.step % self.config.train.checkpoint_every == 0
                        ):
                            self.checkpoint()
                        if (
                            self.config.train.eval_every > 0
                            and self.step % self.config.train.eval_every == 0
                        ):
                            evaluate_views(self.pipeline, self.root, save_renders=False)
                self.checkpoint(final=True)
                self.write_state("TRAINING_COMPLETE")
                evaluate_views(self.pipeline, self.root)
                if self.config.train.export_on_finish:
                    self.write_state("EXPORTING")
                    from .exporters import export_pipeline

                    report = export_pipeline(self.pipeline, self.root)
                    self.write_state("COMPLETE", export=report)
                else:
                    self.write_state("TRAINING_COMPLETE", export_pending=True)
                return self.pipeline
            except (Exception, KeyboardInterrupt) as exc:
                # CUDA device assertions can poison the context, so even RNG
                # capture/checkpointing may fail. Preserve the original cause
                # and record the last durable checkpoint without hiding it.
                checkpoint_error = None
                try:
                    self.checkpoint()
                except Exception as save_error:
                    checkpoint_error = f"{type(save_error).__name__}: {save_error}"
                latest = self.root / "checkpoints" / "latest.pt"
                self.write_state(
                    "FAILED"
                    if not isinstance(exc, KeyboardInterrupt)
                    else "INTERRUPTED",
                    error=f"{type(exc).__name__}: {exc}",
                    recoverable_checkpoint=str(latest) if latest.is_file() else None,
                    checkpoint_save_error=checkpoint_error,
                )
                raise


def load_run(root, *, device=None, checkpoint="latest.pt"):
    from .config import load_config

    root = Path(root).resolve()
    config = load_config(root / "config.yaml")
    if device:
        config.train.device = device
    payload = load_checkpoint(root / "checkpoints" / checkpoint, config.train.device)
    pipeline = methods.get(config.method).make_pipeline(config)
    pipeline.load_state_dict(payload["pipeline"])
    return pipeline, payload["step"]
