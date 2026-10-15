"""Recorded CPU end-to-end verification on the analytic demo dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from splatforge.studio.artifacts import atomic_json, timestamp
from splatforge.studio.config import load_config
from splatforge.studio.demo import make_demo
from splatforge.studio.exporters import export_pipeline
from splatforge.studio.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    data = args.work / "data"
    if not (data / "scene.json").is_file():
        make_demo(data, width=24, height=24, points=48)
    report = {
        "updated": timestamp(),
        "torch": torch.__version__,
        "device": "cpu",
        "verification": "analytic initialized sphere; not a reconstruction benchmark",
        "methods": {},
    }
    for name in ("radegs", "ambisur", "gof"):
        output = args.work / name
        config = load_config(
            method=name,
            overrides=[
                "data.source=" + json.dumps(str(data.resolve())),
                "train.output=" + json.dumps(str(output.resolve())),
                "train.device=cpu",
                "render.backend=torch",
                "data.downscale=1",
                "data.init_points=48",
                "model.opacity=0.7",
                "model.sh_degree_interval=0",
                "train.steps=3",
                "train.export_on_finish=false",
                "train.eval_every=0",
                "train.checkpoint_every=0",
                "densify.enabled=false",
                "loss.geometric_start=0",
                "loss.ambiguity_start=0",
                "export.alpha_min=0.1",
                "export.truncation=0.15",
            ],
        )
        pipeline = Trainer(
            config,
            resume="latest" if (output / "checkpoints/latest.pt").is_file() else None,
        ).run()
        exported = export_pipeline(pipeline, output)
        report["methods"][name] = {
            "status": "PASS",
            "actual_steps": 3,
            "export": exported,
            "operator_revision": pipeline.ops.runtime.definitions.get(
                "gaussian.covariance@1"
            ).spec_revision,
        }
        atomic_json(args.report, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
