"""Unified SDFStudio-style workflow for pluggable surface methods."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parser():
    root = argparse.ArgumentParser(
        prog="splatforge", description="Modular Gaussian surface reconstruction studio"
    )
    commands = root.add_subparsers(dest="command", required=True)
    methods = commands.add_parser(
        "methods", help="list registered methods and their actual backends"
    )
    methods.add_argument("--json", action="store_true")
    train = commands.add_parser("train", help="train or resume a method")
    train.add_argument("method", nargs="?")
    train.add_argument("--config", type=Path)
    train.add_argument("--data", type=Path)
    train.add_argument("--output", type=Path)
    train.add_argument("--device")
    train.add_argument("--backend")
    train.add_argument("--steps", type=int)
    train.add_argument("--resume", nargs="?", const="latest")
    train.add_argument(
        "--set", action="append", default=[], metavar="SECTION.KEY=VALUE"
    )
    configure = commands.add_parser("config", help="write an editable method preset")
    configure.add_argument("method")
    configure.add_argument("--output", type=Path, required=True)
    configure.add_argument("--data", default="DATASET_PATH")
    export = commands.add_parser(
        "export", help="export mesh or Gaussians from a saved run"
    )
    export.add_argument("--run", type=Path, required=True)
    export.add_argument("--device")
    export.add_argument("--kind")
    export.add_argument("--set", action="append", default=[])
    evaluate = commands.add_parser(
        "eval", help="evaluate saved views or explicit geometry inputs"
    )
    evaluate.add_argument("--run", type=Path)
    evaluate.add_argument("--device")
    evaluate.add_argument("--split", default="test", choices=("test", "train"))
    evaluate.add_argument("--prediction", type=Path)
    evaluate.add_argument("--ground-truth", type=Path)
    evaluate.add_argument("--alignment", type=Path)
    evaluate.add_argument("--threshold", type=float, default=0.005)
    evaluate.add_argument("--unit", default="world")
    evaluate.add_argument("--output", type=Path)
    viewer = commands.add_parser(
        "viewer", help="local interactive scene and rendering viewer"
    )
    viewer.add_argument("--run", type=Path, required=True)
    viewer.add_argument("--device")
    viewer.add_argument("--host", default="127.0.0.1")
    viewer.add_argument("--port", type=int, default=8080)
    demo = commands.add_parser(
        "demo", help="generate an analytic sphere scene with metric depth priors"
    )
    demo.add_argument("--output", type=Path, required=True)
    demo.add_argument("--width", type=int, default=32)
    demo.add_argument("--points", type=int, default=96)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    from .config import load_config
    from .methods import methods

    if args.command == "methods":
        entries = []
        for name in methods.names():
            recipe = methods.get(name)
            preset = recipe.default_config()
            entries.append(
                {
                    "name": name,
                    "description": recipe.description,
                    "default_backend": preset.render.backend,
                    "exporter": recipe.exporter,
                    "requires_depth_prior": recipe.requires_prior,
                    "status": "experimental",
                }
            )
        print(
            json.dumps(entries, indent=2)
            if args.json
            else "\n".join(
                f"{item['name']:10} {item['default_backend']:16} {item['description']}"
                for item in entries
            )
        )
    elif args.command == "config":
        config = methods.get(args.method).default_config()
        config.data.source = args.data
        if args.output.exists():
            raise ValueError("config destination already exists")
        config.save(args.output)
        print(args.output)
    elif args.command == "train":
        from .trainer import Trainer

        overrides = list(args.set)
        for name, value in (
            ("data.source", args.data),
            ("train.output", args.output),
            ("train.device", args.device),
            ("render.backend", args.backend),
            ("train.steps", args.steps),
        ):
            if value is not None:
                overrides.append(
                    f"{name}={json.dumps(str(value))}"
                    if isinstance(value, Path)
                    else f"{name}={value}"
                )
        config = load_config(args.config, method=args.method, overrides=overrides)
        Trainer(config, resume=args.resume).run()
    elif args.command == "export":
        from .trainer import load_run
        from .exporters import export_pipeline

        pipeline, _ = load_run(args.run, device=args.device)
        overrides = [f"export.kind={args.kind}"] if args.kind else []
        overrides += args.set
        config = load_config(args.run / "config.yaml", overrides=overrides)
        pipeline.config.export = config.export
        print(json.dumps(export_pipeline(pipeline, args.run), indent=2))
    elif args.command == "eval":
        from .evaluation import evaluate_views, geometry_metrics
        from .artifacts import atomic_json

        if args.ground_truth:
            if not args.prediction:
                raise ValueError("--ground-truth requires --prediction")
            import numpy as np
            from ..mesh.io import read_vertices

            alignment = np.loadtxt(args.alignment) if args.alignment else None
            result = geometry_metrics(
                read_vertices(args.prediction),
                read_vertices(args.ground_truth),
                threshold=args.threshold,
                unit=args.unit,
                transform=alignment,
            )
        else:
            if not args.run:
                raise ValueError("eval needs --run or --prediction with --ground-truth")
            from .trainer import load_run

            pipeline, _ = load_run(args.run, device=args.device)
            result = evaluate_views(pipeline, args.run, split=args.split)
        if args.output:
            atomic_json(args.output, result)
        print(json.dumps(result, indent=2))
    elif args.command == "viewer":
        from .viewer import serve

        serve(args.run, host=args.host, port=args.port, device=args.device)
    elif args.command == "demo":
        from .demo import make_demo

        print(
            make_demo(
                args.output, width=args.width, height=args.width, points=args.points
            )
        )


def train_main():
    return main(["train", *sys.argv[1:]])


def export_main():
    return main(["export", *sys.argv[1:]])


def eval_main():
    return main(["eval", *sys.argv[1:]])


def viewer_main():
    return main(["viewer", *sys.argv[1:]])


if __name__ == "__main__":
    main()
