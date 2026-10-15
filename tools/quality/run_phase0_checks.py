"""Run the complete local Phase 0 judge-system verification sequence."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import jsonschema


def _run(repo: Path, command: list[str], env: dict[str, str]) -> bool:
    print("RUN", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=repo, env=env, check=False)
    print(f"EXIT {completed.returncode}", flush=True)
    return completed.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("benchmarks/results/phase0/phase0_conformance.local.json"),
    )
    args = parser.parse_args()

    repo = args.repo.resolve()
    report = args.report if args.report.is_absolute() else repo / args.report
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    entries = [str(repo / "src"), str(repo)]
    if existing:
        entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(entries)

    commands = [
        [sys.executable, "-m", "compileall", "-q", "src", "tools", "tests"],
        [sys.executable, "tools/quality/validate_manifests.py", "--repo", str(repo)],
        [sys.executable, "-m", "pytest", "tests", "-q"],
        [
            sys.executable,
            "-m",
            "splatforge.testing.phase0_demo",
            "--repo",
            str(repo),
            "--output",
            str(report),
        ],
    ]
    if not all(_run(repo, command, env) for command in commands):
        return 1

    schema = json.loads(
        (repo / "tools/schemas/evidence_report.schema.json").read_text(
            encoding="utf-8"
        )
    )
    value = json.loads(report.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(
        schema,
        format_checker=jsonschema.FormatChecker(),
    ).validate(value)
    if value["status"] != "passed":
        print("Phase 0 conformance report did not pass", flush=True)
        return 1
    print(f"PHASE0_LOCAL_CHECKS_PASSED report={report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

