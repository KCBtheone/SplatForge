"""Reusable training engine with exact optimizer/RNG/sampler checkpoint state."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from uuid import uuid4

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
from .._paths import output_path
from .resources import wait_for_capacity
from .step import StepContext, training_step


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
        if not record["loss"].get("radegs_normalize_numerator", False):
            record["loss"].pop("radegs_normalize_numerator", None)
        # Absence of the new disabled option has the original semantics.
        # Enabling it still requires its own model/network/optimizer checkpoint.
        if not record["model"].get("paper_decoupled_appearance",False):
            record["model"].pop("paper_decoupled_appearance",None)
        if not record["model"].get("paper_sh_lr_split",False):
            record["model"].pop("paper_sh_lr_split",None)
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


def optimizer_layout(pipeline, optimizer):
    named_parameters = getattr(pipeline.model, "named_parameters", tuple)
    names = {id(parameter): name for name, parameter in named_parameters()}
    seen = set()
    layout = []
    for group in optimizer.param_groups:
        parameters = []
        for parameter in group["params"]:
            identity = id(parameter)
            if identity in seen:
                raise ValueError("an optimizer parameter is registered more than once")
            seen.add(identity)
            parameters.append([names.get(identity), list(parameter.shape)])
        layout.append([group.get("name"), parameters])
    return layout


def restore_optimizer(pipeline, optimizer, checkpoint):
    saved = checkpoint["optimizer"]
    current_layout = optimizer_layout(pipeline, optimizer)
    stored_layout = checkpoint.get("optimizer_layout")
    if stored_layout is not None and stored_layout != current_layout:
        raise ValueError("checkpoint optimizer parameter names or layout changed")
    saved_groups = saved["param_groups"]
    if [(group.get("name"), len(group["params"])) for group in saved_groups] != [
        (name, len(parameters)) for name, parameters in current_layout
    ]:
        raise ValueError("checkpoint optimizer groups changed")
    for group, (_, parameters) in zip(saved_groups, current_layout):
        for key, (_, shape) in zip(group["params"], parameters):
            for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                moment = saved["state"].get(key, {}).get(name)
                if moment is not None and list(moment.shape) != shape:
                    raise ValueError("checkpoint Adam moment does not match parameter shape")
    optimizer.load_state_dict(saved)


class Trainer:
    def __init__(self, config, *, resume=None):
        self.config = config.validate()
        self.root = output_path(config.train.output)
        self.resume = resume
        self.pipeline = None
        self.optimizer = None
        self.step = 0
        self.order = []
        self.order_offset = 0
        self.manifest = None
        self.next_stage = "train"
        self._training_checkpoint = None
        self._uncommitted_update = False

    def _read_stage_progress(self):
        try:
            return json.loads((self.root / "state.json").read_text(encoding="utf-8")).get(
                "progress", {}
            )
        except (FileNotFoundError, ValueError):
            return {}

    def _restore_stage_progress(self, checkpoint):
        progress = checkpoint.get("progress", {})
        if self.step < self.config.train.steps or not progress.get("training_checkpoint"):
            return
        self._training_checkpoint = progress["training_checkpoint"]
        self.next_stage = "evaluate"
        # Finalization does not modify the saved training state. The small
        # atomic run record is bound to that checkpoint, avoiding another
        # serialization of the model and Adam after evaluation and export.
        recorded = self._read_stage_progress()
        if recorded.get("training_checkpoint") != self._training_checkpoint:
            return
        if recorded.get("next_stage") in ("export", "complete"):
            self.next_stage = "export"
        if (
            recorded.get("next_stage") == "complete"
            and recorded.get("export_config") == self.config.to_dict()["export"]
            and (self.root / "export.json").is_file()
        ):
            self.next_stage = "complete"

    def write_state(self, stage, **extra):
        progress = {
            "next_stage": self.next_stage,
            "training_checkpoint": self._training_checkpoint,
        }
        if self.next_stage == "complete":
            progress["export_config"] = self.config.to_dict()["export"]
        if self.pipeline is None and self.resume:
            # A capacity wait before setup must preserve finalization progress.
            progress = self._read_stage_progress() or progress
        atomic_json(
            self.root / "state.json",
            {
                "stage": stage,
                "step": self.step,
                "method": self.config.method,
                "updated": timestamp(),
                "progress": progress,
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
            restore_optimizer(self.pipeline, self.optimizer, checkpoint)
            restore_random_state(checkpoint["random"])
            self._restore_stage_progress(checkpoint)
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
        if self.next_stage == "train":
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
            "optimizer_layout": optimizer_layout(self.pipeline, self.optimizer),
            "random": random_state(),
            "sampler": {"order": self.order, "offset": self.order_offset},
            "progress": {
                "next_stage": self.next_stage,
                "training_checkpoint": self._training_checkpoint,
            },
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

    def _begin_update(self):
        self._uncommitted_update = True

    def _sample(self):
        return self.pipeline.data.sample(self.sample_index(), self.config.train.device)

    def _train(self):
        settings = self.config.train
        last_disk_check = time.monotonic()
        step = getattr(getattr(self.pipeline, "method", None), "training_step", training_step)
        with (self.root / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            while self.step < settings.steps:
                if time.monotonic() - last_disk_check >= 60:
                    self.checkpoint()
                    wait_for_capacity(self.config, self.write_state, check_gpu=False)
                    self.write_state("TRAINING")
                    last_disk_check = time.monotonic()
                # An overridden step may update persistent state at any point.
                self._uncommitted_update = step is not training_step
                result = step(StepContext(
                    pipeline=self.pipeline,
                    optimizer=self.optimizer,
                    iteration=self.step,
                    sample=self._sample,
                    check_gradients=getattr(self, "gradient_finite_check", None),
                    begin_update=self._begin_update,
                ))
                self.order_offset += 1
                self.step += 1
                self._uncommitted_update = False
                if self.step == 1 or self.step % settings.log_every == 0 or self.step == settings.steps:
                    row = {
                        "step": self.step,
                        "loss": float(result.loss),
                        "terms": {key: float(value) for key, value in result.terms.items()},
                        **self.pipeline.statistics(),
                        "time": timestamp(),
                    }
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
                    handle.flush()
                    self.write_state("TRAINING", loss=row["loss"])
                    print(json.dumps(row, allow_nan=False), flush=True)
                if settings.checkpoint_every > 0 and self.step % settings.checkpoint_every == 0:
                    self.checkpoint()
                if settings.eval_every > 0 and self.step % settings.eval_every == 0:
                    evaluate_views(self.pipeline, self.root, save_renders=False)
        self.next_stage = "evaluate"
        self._training_checkpoint = uuid4().hex
        self.checkpoint(final=True)
        self.write_state("TRAINING_COMPLETE")

    def _finalize(self):
        if self.next_stage == "evaluate":
            evaluate_views(self.pipeline, self.root)
            self.next_stage = "export"
            self.write_state("TRAINING_COMPLETE", export_pending=True)
        if self.next_stage == "export" and self.config.train.export_on_finish:
            from .exporters import export_pipeline

            self.write_state("EXPORTING")
            report = export_pipeline(self.pipeline, self.root)
            self.next_stage = "complete"
            self.write_state("COMPLETE", export=report)

    def run(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with RunLock(self.root):
            if (self.root / "checkpoints" / "latest.pt").exists() and not self.resume:
                raise ValueError(
                    "run already has a checkpoint; pass --resume latest or choose a new output"
                )
            wait_for_capacity(self.config, self.write_state)
            self.setup()
            try:
                if self.next_stage == "train":
                    self._train()
                self._finalize()
                return self.pipeline
            except (Exception, KeyboardInterrupt) as exc:
                # A failed optimizer/topology update may have changed only part
                # of the representation. Keep the last complete checkpoint.
                checkpoint_error = None
                if self.next_stage == "train" and not self._uncommitted_update:
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
    recipe = methods.get(config.method)
    if hasattr(recipe, "export_run"):
        raise ValueError(f"{config.method} uses native checkpoints. Use sf-export --run {root} for geometry.")
    if device:
        config.train.device = device
    payload = load_checkpoint(root / "checkpoints" / checkpoint, config.train.device)
    pipeline = recipe.make_pipeline(config)
    pipeline.load_state_dict(payload["pipeline"])
    return pipeline, payload["step"]
