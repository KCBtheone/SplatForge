"""Capture a non-secret, machine-readable runtime/toolchain fingerprint."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jsonschema
import torch

from splatforge.core.digests import canonical_json_digest


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _gpus() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,compute_cap,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError:
        # CPU-only hosts need not install NVIDIA management utilities.
        return []
    if completed.returncode != 0:
        return []
    records: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        records.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "compute_capability": parts[2],
                "memory_total_mib": int(parts[3]),
                "driver": parts[4],
            }
        )
    return records


def capture() -> dict[str, Any]:
    facts = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "packages": {
            name: _package_version(name)
            for name in ("torch", "triton", "PyYAML", "jsonschema")
        },
        "torch": {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cudnn": torch.backends.cudnn.version(),
        },
        "gpus": _gpus(),
    }
    return {
        "schema_version": 1,
        "authority": "self_reported",
        "captured_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        **facts,
        "environment_id": canonical_json_digest(facts),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    value = capture()
    schema = json.loads(
        (args.repo.resolve() / "tools/schemas/environment_lock.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.Draft202012Validator(
        schema,
        format_checker=jsonschema.FormatChecker(),
    ).validate(value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"environment_id": value["environment_id"], "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

