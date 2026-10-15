"""Exercise installed console scripts and the README's analytic CPU workflow."""

from __future__ import annotations

import argparse
import importlib.resources
import json
import os
import subprocess
import sys
import sysconfig
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=False)
    import torch
    import splatforge
    from splatforge.gaussian import SurfaceOperators

    installed = Path(sysconfig.get_paths()["purelib"]).resolve()
    if not Path(splatforge.__file__).resolve().is_relative_to(installed):
        raise RuntimeError("Verify a wheel installed in this environment, not a source checkout")
    for package, names in {
        "splatforge.gaussian": ["semantics.json"],
        "splatforge.providers.torch.gaussian": ["composite_cuda.cu"],
        "splatforge.providers.torch.testing": ["scale_shift.implementation.yaml"],
        "splatforge.studio": ["viewer.html", "viewer.js", "viewer.css"],
    }.items():
        for name in names:
            if not importlib.resources.files(package).joinpath(name).is_file():
                raise RuntimeError(f"Missing package resource: {package}/{name}")
    q = torch.tensor([[1., 0., 0., 0.]], requires_grad=True)
    s = torch.tensor([[.1, .2, .3]], requires_grad=True)
    SurfaceOperators()("gaussian.covariance@1", q, s).square().sum().backward()
    if not torch.isfinite(s.grad).all():
        raise RuntimeError("Operator example produced invalid gradients")

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    scripts = Path(sysconfig.get_paths()["scripts"])
    suffix = ".exe" if os.name == "nt" else ""
    checks = []

    def run(name, *arguments):
        command = [str(scripts / (name + suffix)), *map(str, arguments)]
        result = subprocess.run(
            command, cwd=work, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        (work / f"{len(checks):02}-{name}.log").write_text(result.stdout, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"Installed {name} failed; inspect the verification logs")
        checks.append(name)
        return result.stdout

    for name in ("splatforge", "sf-train", "sf-export", "sf-eval", "sf-viewer"):
        run(name, "--help")
    methods = json.loads(run("splatforge", "methods", "--json"))
    names = sorted(row["name"] for row in methods)
    if names != ["ambisur", "gof", "radegs"]:
        raise RuntimeError("Unexpected preview method registry")
    data, output = work / "data", work / "run"
    run("splatforge", "demo", "--output", data, "--width", 24, "--points", 48)
    run(
        "sf-train", "gof", "--data", data, "--output", output,
        "--device", "cpu", "--backend", "torch", "--steps", 3,
        "--set", "data.downscale=1", "--set", "data.init_points=48",
        "--set", "model.opacity=0.7", "--set", "densify.enabled=false",
        "--set", "loss.geometric_start=0", "--set", "train.export_on_finish=false",
    )
    run(
        "sf-export", "--run", output, "--device", "cpu",
        "--set", "export.alpha_min=0.1", "--set", "export.truncation=0.15",
    )
    run("sf-eval", "--run", output, "--device", "cpu")
    if not (output / "mesh.ply").is_file():
        raise RuntimeError("Installed workflow did not export a mesh")
    report = {
        "status": "PASS", "version": splatforge.__version__,
        "python": sys.version.split()[0], "torch": torch.__version__,
        "device": "cpu", "methods": names, "console_invocations": len(checks),
        "packaged_resources": "PASS", "operator_gradient": "PASS",
        "analytic_demo_train_export_evaluate": "PASS",
        "claim_boundary": "synthetic workflow check; no paper or GPU performance claim",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
