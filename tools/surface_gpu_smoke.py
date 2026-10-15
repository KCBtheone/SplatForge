"""Actual CUDA train/export checks; synthetic verification is not a benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from splatforge.studio.artifacts import atomic_json
from splatforge.studio.config import load_config
from splatforge.studio.demo import make_demo
from splatforge.studio.exporters import export_pipeline
from splatforge.studio.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable; this check cannot be replaced by a CPU result"
        )
    torch.set_num_threads(2)
    data = args.work / "data"
    if not (data / "scene.json").is_file():
        make_demo(data, width=24, height=24, points=48)
    report = {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "verification": "synthetic actual CUDA; not paper benchmark parity",
        "methods": {},
    }
    for name in ("gof", "ambisur", "radegs"):
        root = args.work / name
        overrides = [
            "data.source=" + json.dumps(str(data)),
            "train.output=" + json.dumps(str(root)),
            "train.device=cuda",
            "render.backend=" + ("gsplat-radegs" if name == "radegs" else "torch"),
            "data.downscale=1",
            "data.init_points=48",
            "model.opacity=0.7",
            "train.steps=3",
            "train.export_on_finish=false",
            "train.eval_every=0",
            "train.checkpoint_every=0",
            "densify.enabled=false",
            "loss.geometric_start=0",
            "loss.ambiguity_start=0",
            "export.alpha_min=0.1",
            "export.truncation=0.15",
        ]
        config = load_config(method=name, overrides=overrides)
        try:
            pipeline = Trainer(
                config,
                resume="latest" if (root / "checkpoints/latest.pt").is_file() else None,
            ).run()
            result = export_pipeline(pipeline, root)
            report["methods"][name] = {
                "status": "PASS",
                "actual_steps": 3,
                "backend": config.render.backend,
                "export": result,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            }
        except Exception as exc:
            report["methods"][name] = {
                "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
            }
            atomic_json(args.output, report)
            raise
        atomic_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
