"""Copy an explicit source allowlist into a new public-release directory.

The development workspace also contains independent experiments and private
run artifacts. Never use a recursive copy of that workspace for publication.
This script only prepares files; it does not commit, upload, or delete them.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


ROOT_FILES = (
    ".gitattributes", ".gitignore", "ARCHITECTURE.md", "CONTRIBUTING.md",
    "LICENSE", "NOTICE", "README.md", "REUSE.toml", "THIRD_PARTY.md",
    "MANIFEST.in", "pyproject.toml", "docs/ROADMAP.md",
    "docs/studio/PUBLIC_GUIDE.md", "tools/__init__.py",
    "tools/surface_cpu_smoke.py", "tools/surface_gpu_smoke.py",
)
TREES = (
    "src", "specs", "provenance", "tests", "tools/quality", "tools/schemas",
    "tools/release", "docs/adr", "docs/reimplementations/pgsr", "docs/releases",
    "work_orders/drafts", ".github/workflows",
)
LOCAL_TESTS = {
    "tests/integration/test_object_status.py",
    "tests/integration/test_object_suite.py",
    "tests/integration/test_surface_status.py",
    "tests/integration/test_surface_live_inspection.py",
    "tests/integration/test_surface_performance_probe.py",
}
COMPANION_FILES = (
    "AGENTS.md", "LICENSE", "NOTICE", "SOURCE_POLICY.md", "SPEC.md",
    "pyproject.toml", "setup.py",
)
COMPANION_LOCAL_TESTS = {
    "test_coords_dataloader.py", "test_geometry_export.py",
    "test_license_evidence.py", "test_real_checkpoint_benchmark.py",
}
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache"}
SKIP_SUFFIXES = {".pyc", ".pyo"}
SECRETS = re.compile(
    r"-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----"
    r"|(?:gh[pousr]_|github_pat_|hf_)[A-Za-z0-9_]{24,}"
    r"|https?://[^\s/:@]+:[^\s/@]+@"
)


def copy_file(source: Path, target: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"Expected a regular source file: {source.name}")
    if source.stat().st_size > 2 * 1024 * 1024:
        raise ValueError(f"Unexpected large source file: {source.name}")
    content = source.read_bytes()
    if SECRETS.search(content.decode("utf-8")):
        raise ValueError(f"Potential credential in source file: {source.name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def copy_tree(source: Path, target: Path, *, excludes=()) -> None:
    for entry in sorted(source.rglob("*")):
        relative = entry.relative_to(source)
        if (
            SKIP_PARTS.intersection(relative.parts)
            or any(part.endswith(".egg-info") for part in relative.parts)
            or entry.suffix in SKIP_SUFFIXES
        ):
            continue
        if relative.as_posix() in excludes or entry.name in excludes:
            continue
        if entry.is_symlink():
            raise ValueError(f"Symlink in public source: {relative}")
        if entry.is_file():
            copy_file(entry, target / relative)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    destination = args.destination.resolve()
    if not destination.is_relative_to(repo / "build"):
        raise ValueError("Prepare public snapshots inside the workspace build directory")
    if destination.exists():
        raise ValueError("Destination already exists; use a new snapshot directory")
    destination.mkdir(parents=True)

    for relative in ROOT_FILES:
        copy_file(repo / relative, destination / relative)
    for relative in TREES:
        excluded = (
            {path.removeprefix("tests/") for path in LOCAL_TESTS}
            if relative == "tests" else ()
        )
        copy_tree(repo / relative, destination / relative, excludes=excluded)

    companion = "commercial_unbiased_gsplat"
    for relative in COMPANION_FILES:
        copy_file(repo / companion / relative, destination / companion / relative)
    for relative in ("unbiased_gsplat", "csrc", "tests"):
        copy_tree(
            repo / companion / relative, destination / companion / relative,
            excludes=COMPANION_LOCAL_TESTS if relative == "tests" else (),
        )
    copy_file(
        repo / "docs/releases/unbiased-preview.md",
        destination / companion / "README.md",
    )
    (destination / companion / "MANIFEST.in").write_text(
        "include LICENSE NOTICE README.md SOURCE_POLICY.md SPEC.md\n"
        "recursive-include csrc *.cu *.cpp\n"
        "recursive-include tests *.py\n"
        "global-exclude __pycache__ *.py[cod]\n",
        encoding="utf-8",
    )
    version = re.search(
        r'^version = "([^"]+)"$',
        (repo / "pyproject.toml").read_text(encoding="utf-8"),
        re.MULTILINE,
    ).group(1)
    files = [path for path in destination.rglob("*") if path.is_file()]
    print(json.dumps({
        "version": version,
        "files": len(files),
        "bytes": sum(path.stat().st_size for path in files),
        "destination": str(destination),
        "scope": "public source allowlist; no local run artifacts or development history",
    }, indent=2))


if __name__ == "__main__":
    main()
