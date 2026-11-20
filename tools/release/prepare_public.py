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
    ".gitattributes", ".gitignore", "CONTRIBUTING.md",
    "LICENSE", "NOTICE", "README.md", "REUSE.toml", "THIRD_PARTY.md",
    "MANIFEST.in", "setup.py", "docs/ROADMAP.md",
    "docs/adr/0004-modular-surface-studio.md", "docs/mesh/PROCESSING.md",
    "docs/studio/PUBLIC_GUIDE.md", "provenance/surface_methods.json",
    "provenance/testing.scale_shift.yaml", "specs/catalog.json", "specs/mesh/SEMANTICS.md",
    "tests/__init__.py",
    "tools/__init__.py", "tools/build_geometry.py", "tools/build_geometry_cuda.py",
    "tools/generate_ray_geometry_reverse.py", "tools/geometry_benchmark.py",
    "tools/surface_cpu_smoke.py", "tools/surface_gpu_smoke.py",
    "tools/surface_cuda_binning_check.py", "tools/surface_cuda_events_check.py",
    "tools/surface_ray_geometry_cuda_check.py",
    ".github/workflows/ci.yml",
)
TREES = (
    "src/splatforge", "specs/testing", "tests/architecture", "tests/conformance",
    "tests/core", "tests/integration", "tests/quality", "tools/quality",
    "tools/schemas", "tools/release", "docs/releases",
)
LOCAL_TESTS = {
    "test_cuda_high_count_preflight.py",
    "test_object_status.py",
    "test_object_suite.py",
    "test_surface_acceleration_binding.py",
    "test_surface_dtu_benchmark.py",
    "test_surface_dtu_trial.py",
    "test_surface_live_inspection.py",
    "test_surface_performance_probe.py",
    "test_surface_status.py",
}
COMPANION_FILES = (
    "LICENSE", "NOTICE", "SOURCE_POLICY.md",
    "pyproject.toml", "setup.py",
    "csrc/unbiased_fused_bind.cpp", "csrc/unbiased_fused.cu",
    "csrc/unbiased_aux.cu", "csrc/rasterize_unbiased_aux.cu",
)
COMPANION_LOCAL_TESTS = {
    "test_coords_dataloader.py", "test_geometry_export.py",
    "test_license_evidence.py", "test_real_checkpoint_benchmark.py",
}
PGSR_ROOT = "self_reports/pgsr_standalone_20261019/source"
NATIVE_SOURCE = "src/splatforge/studio/native_source"
NATIVE_MODULES = (
    "__init__", "_cuda", "_validation", "adjoint_tile_execution", "background",
    "cached_calibration", "commercial_model", "convergence", "corrected_depth",
    "depth_normal", "dual_geometry", "family_roles", "gradient_validation",
    "hit_sequence", "input_photometric_consensus", "losses", "models", "multiview",
    "native_multiview", "normal_support", "optimizer_execution", "photometric",
    "physical_ordering", "physical_refinement", "planar_gaussian", "ray_consensus",
    "ray_splat_depth", "refinement_schedule", "regional_commercial",
    "regional_identity_consistency", "regional_image_fusion", "regional_image_graph",
    "sfm_orientation", "sfm_surface_anchor", "shared_surfel_execution",
    "surfel_execution", "trainer_adapter", "training_state", "view_tensor_cache",
)
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".pyd", ".so", ".dylib", ".dll", ".lib", ".obj", ".exp"}
SECRETS = re.compile(
    r"-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----"
    r"|(?:gh[pousr]_|github_pat_|hf_)[A-Za-z0-9_]{24,}"
    r"|https?://[^\s/:@]+:[^\s/@]+@"
)


def copy_file(source: Path, target: Path, *, before: str = "", omit_prefixes=()) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"Expected a regular source file: {source.name}")
    if source.stat().st_size > 2 * 1024 * 1024:
        raise ValueError(f"Unexpected large source file: {source.name}")
    content = source.read_text(encoding="utf-8")
    if before:
        content = content.split(before, 1)[0].rstrip() + "\n"
    if omit_prefixes:
        content = "".join(
            line for line in content.splitlines(keepends=True)
            if not line.startswith(omit_prefixes)
        )
    if SECRETS.search(content):
        raise ValueError(f"Potential credential in source file: {source.name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if before or omit_prefixes:
        target.write_text(content, encoding="utf-8")
    else:
        shutil.copyfile(source, target)


def copy_tree(source: Path, target: Path, *, excludes=()) -> None:
    for entry in sorted(source.rglob("*")):
        relative = entry.relative_to(source)
        if (
            SKIP_PARTS.intersection(relative.parts)
            or any(part.endswith(".egg-info") for part in relative.parts)
            or entry.suffix in SKIP_SUFFIXES
            or entry.name.endswith(".local.json")
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
    parser.add_argument("--pgsr-source", type=Path, help="Override the shared PGSR and Surfel source directory")
    args = parser.parse_args()
    repo = args.repo.resolve()
    destination = args.destination.resolve()
    if not destination.is_relative_to(repo / "build"):
        raise ValueError("Prepare public snapshots inside the workspace build directory")
    if destination.exists():
        raise ValueError("Destination already exists; use a new snapshot directory")

    pgsr_source = repo / NATIVE_SOURCE
    pgsr_entry = pgsr_source / "train.py"
    if not pgsr_entry.is_file():
        pgsr_source = repo / PGSR_ROOT
        pgsr_entry = repo / "deploy/runpod/pgsr.py"
    if args.pgsr_source is not None:
        pgsr_source = args.pgsr_source.expanduser().resolve()
    pgsr_files = (
        "LICENSE", "NOTICE", "tools/commercial_colmap.py", "tools/train_commercial_methods.py",
    ) + tuple(
        f"unbiased_gsplat/{name}.py" for name in NATIVE_MODULES
    )
    missing_pgsr = [
        path for path in (pgsr_entry, *(pgsr_source / name for name in pgsr_files))
        if not path.is_file()
    ]
    if missing_pgsr:
        raise FileNotFoundError(f"Missing native method release input: {missing_pgsr[0]}")
    destination.mkdir(parents=True)

    for relative in ROOT_FILES:
        copy_file(repo / relative, destination / relative)
    copy_file(
        repo / "pyproject.toml", destination / "pyproject.toml",
        omit_prefixes=("research-campaign =", '"research_engine" ='),
    )
    for relative in TREES:
        excluded = LOCAL_TESTS if relative == "tests/integration" else ()
        copy_tree(repo / relative, destination / relative, excludes=excluded)

    companion = "commercial_unbiased_gsplat"
    for relative in COMPANION_FILES:
        copy_file(repo / companion / relative, destination / companion / relative)
    copy_file(
        repo / companion / "SPEC.md", destination / companion / "SPEC.md",
        before="\n## Experiment notes\n",
    )
    for relative in ("unbiased_gsplat", "tests"):
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

    native = destination / NATIVE_SOURCE
    for relative in pgsr_files:
        copy_file(pgsr_source / relative, native / relative)
    copy_file(pgsr_entry, native / "train.py")
    for relative in COMPANION_FILES:
        if relative.startswith("csrc/"):
            copy_file(repo / companion / relative, native / relative)
    copy_file(repo / companion / "NOTICE", native / "csrc/NOTICE")
    pgsr = destination / "pgsr"
    copy_file(repo / "docs/releases/pgsr-preview.md", pgsr / "README.md")
    (pgsr / "train.py").write_text(
        "from pathlib import Path\n"
        "from runpy import run_path\n\n"
        "if __name__ == \"__main__\":\n"
        f"    entry = Path(__file__).resolve().parents[1] / {NATIVE_SOURCE!r} / \"train.py\"\n"
        "    run_path(str(entry), run_name=\"__main__\")\n",
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
