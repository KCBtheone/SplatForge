"""Repository manifest validation and cross-document consistency checks."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema
import yaml

from splatforge.core.digests import hash_path


_PLACEHOLDER_PATTERN = re.compile(
    r"(?:^|[^A-Z0-9])(?:TBD|TO_BE_FILLED|TO_BE_LOCKED|TO_BE_PINNED|"
    r"TO_BE_ARCHIVED|MUTABLE_MAIN)(?:[^A-Z0-9]|$)",
    re.IGNORECASE,
)
_GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    path: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "message": self.message}


def _load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _schema(root: Path, name: str) -> dict[str, Any]:
    value = _load_json(root / "tools" / "schemas" / name)
    if not isinstance(value, dict):
        raise TypeError(f"schema root must be an object: {name}")
    return value


def _schema_issues(
    value: Any,
    schema: dict[str, Any],
    path: Path,
    root: Path,
) -> list[ValidationIssue]:
    validator = jsonschema.Draft202012Validator(
        schema,
        format_checker=jsonschema.FormatChecker(),
    )
    issues: list[ValidationIssue] = []
    for error in sorted(validator.iter_errors(value), key=lambda item: list(item.path)):
        location = ".".join(str(item) for item in error.absolute_path)
        suffix = f" ({location})" if location else ""
        issues.append(
            ValidationIssue(
                path=path.relative_to(root).as_posix(),
                message=error.message + suffix,
            )
        )
    return issues


def find_placeholders(value: Any, location: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, str) and _PLACEHOLDER_PATTERN.search(value):
        found.append(location)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(find_placeholders(item, f"{location}[{index}]"))
    elif isinstance(value, dict):
        for key, item in value.items():
            found.extend(find_placeholders(item, f"{location}.{key}"))
    return found


def validate_work_order_semantics(
    value: dict[str, Any],
    display_path: str,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if value.get("status") != "frozen":
        return issues

    placeholders = find_placeholders(value)
    if placeholders:
        issues.append(
            ValidationIssue(
                display_path,
                "frozen work order contains placeholders at " + ", ".join(placeholders),
            )
        )
    if _GIT_COMMIT_PATTERN.fullmatch(str(value.get("base_commit", ""))) is None:
        issues.append(ValidationIssue(display_path, "frozen base_commit is not an exact Git object ID"))
    for key in ("packet_digest", "op_spec_revision"):
        if _DIGEST_PATTERN.fullmatch(str(value.get(key, ""))) is None:
            issues.append(ValidationIssue(display_path, f"frozen {key} is not a SHA-256 digest"))
    for name, frozen_input in value.get("frozen_inputs", {}).items():
        if _DIGEST_PATTERN.fullmatch(str(frozen_input.get("digest", ""))) is None:
            issues.append(
                ValidationIssue(
                    display_path,
                    f"frozen input {name!r} does not have a SHA-256 digest",
                )
            )
    return issues


def _validate_frozen_numeric_artifact(
    value: dict[str, Any],
    display_path: str,
    *,
    kind: str,
) -> list[ValidationIssue]:
    if value.get("status") != "frozen":
        return []
    issues: list[ValidationIssue] = []
    placeholders = find_placeholders(value)
    if placeholders:
        issues.append(
            ValidationIssue(
                display_path,
                f"frozen {kind} contains placeholders at " + ", ".join(placeholders),
            )
        )
    if _DIGEST_PATTERN.fullmatch(str(value.get("spec_revision", ""))) is None:
        issues.append(
            ValidationIssue(display_path, f"frozen {kind} has no exact spec revision")
        )
    if kind == "tolerance table":
        calibration = value.get("calibration", {})
        if calibration.get("status") != "measured":
            issues.append(
                ValidationIssue(
                    display_path,
                    "frozen tolerance table requires measured calibration",
                )
            )
        for key in ("environment_id", "evidence_digest"):
            if _DIGEST_PATTERN.fullmatch(str(calibration.get(key, ""))) is None:
                issues.append(
                    ValidationIssue(
                        display_path,
                        f"frozen tolerance calibration requires exact {key}",
                    )
                )
    elif kind == "benchmark definition":
        if value.get("thresholds", {}).get("status") != "frozen":
            issues.append(
                ValidationIssue(
                    display_path,
                    "frozen benchmark definition requires frozen thresholds",
                )
            )
        if _DIGEST_PATTERN.fullmatch(
            str(value.get("baseline_environment_id", ""))
        ) is None:
            issues.append(
                ValidationIssue(
                    display_path,
                    "frozen benchmark definition requires an exact baseline environment",
                )
            )
        if not value.get("noise_rule"):
            issues.append(
                ValidationIssue(
                    display_path,
                    "frozen benchmark definition requires a noise rule",
                )
            )
    return issues


def validate_repository(root: str | Path) -> list[ValidationIssue]:
    repo_root = Path(root).resolve()
    issues: list[ValidationIssue] = []

    for path in sorted((repo_root / "tools" / "schemas").glob("*.schema.json")):
        try:
            schema_value = _load_json(path)
            validator_class = jsonschema.validators.validator_for(schema_value)
            validator_class.check_schema(schema_value)
        except (json.JSONDecodeError, jsonschema.SchemaError, OSError) as exc:
            issues.append(
                ValidationIssue(
                    path.relative_to(repo_root).as_posix(),
                    f"invalid JSON Schema: {exc}",
                )
            )

    opspec_schema = _schema(repo_root, "opspec.schema.json")
    spec_catalog_schema = _schema(repo_root, "spec_catalog.schema.json")
    implementation_schema = _schema(repo_root, "implementation_manifest.schema.json")
    work_order_schema = _schema(repo_root, "work_order.schema.json")
    provenance_schema = _schema(repo_root, "provenance.schema.json")
    tolerance_schema = _schema(repo_root, "tolerance_table.schema.json")
    benchmark_definition_schema = _schema(
        repo_root, "benchmark_definition.schema.json"
    )
    benchmark_baseline_schema = _schema(repo_root, "benchmark_baseline.schema.json")
    environment_lock_schema = _schema(repo_root, "environment_lock.schema.json")

    catalog_path = repo_root / "specs" / "catalog.json"
    catalog = _load_json(catalog_path)
    issues.extend(
        _schema_issues(catalog, spec_catalog_schema, catalog_path, repo_root)
    )
    definitions = catalog.get("definitions", {})
    if catalog.get("schema_version") != 1 or not isinstance(definitions, dict):
        issues.append(ValidationIssue("specs/catalog.json", "invalid definition catalog envelope"))
        definitions = {}

    seen_specs: set[str] = set()
    for path in sorted((repo_root / "specs").glob("**/opspec.yaml")):
        value = _load_yaml(path)
        issues.extend(_schema_issues(value, opspec_schema, path, repo_root))
        if not isinstance(value, dict) or "id" not in value:
            continue
        op_id = str(value["id"])
        seen_specs.add(op_id)
        entry = definitions.get(op_id)
        if not isinstance(entry, dict):
            issues.append(
                ValidationIssue(path.relative_to(repo_root).as_posix(), "operator is missing from specs/catalog.json")
            )
            continue
        package_path = repo_root / str(entry.get("package", ""))
        if package_path.resolve() != path.parent.resolve():
            issues.append(
                ValidationIssue(path.relative_to(repo_root).as_posix(), "catalog package does not name this spec package")
            )
            continue
        actual = hash_path(package_path)
        if entry.get("spec_revision") != actual:
            issues.append(
                ValidationIssue(
                    "specs/catalog.json",
                    f"{op_id} digest mismatch: catalog={entry.get('spec_revision')} actual={actual}",
                )
            )

    for op_id in sorted(set(definitions) - seen_specs):
        issues.append(ValidationIssue("specs/catalog.json", f"catalog entry has no OpSpec: {op_id}"))

    for path in sorted(repo_root.glob("src/splatforge/providers/**/*.implementation.yaml")):
        value = _load_yaml(path)
        issues.extend(_schema_issues(value, implementation_schema, path, repo_root))
        if not isinstance(value, dict):
            continue
        op_id = str(value.get("op", ""))
        expected = definitions.get(op_id, {}).get("spec_revision")
        if expected is None:
            issues.append(
                ValidationIssue(path.relative_to(repo_root).as_posix(), f"implementation references unknown operator {op_id!r}")
            )
        elif value.get("op_spec_revision") != expected:
            issues.append(
                ValidationIssue(path.relative_to(repo_root).as_posix(), "implementation spec revision does not match catalog")
            )

    for path in sorted((repo_root / "work_orders").glob("**/*.yaml")):
        value = _load_yaml(path)
        issues.extend(_schema_issues(value, work_order_schema, path, repo_root))
        if isinstance(value, dict):
            issues.extend(
                validate_work_order_semantics(
                    value,
                    path.relative_to(repo_root).as_posix(),
                )
            )

    for path in sorted((repo_root / "provenance").glob("*.yaml")):
        value = _load_yaml(path)
        issues.extend(_schema_issues(value, provenance_schema, path, repo_root))

    tolerance_root = repo_root / "tests" / "tolerances"
    if tolerance_root.is_dir():
        for path in sorted(tolerance_root.glob("*.yaml")):
            value = _load_yaml(path)
            issues.extend(_schema_issues(value, tolerance_schema, path, repo_root))
            if isinstance(value, dict):
                issues.extend(
                    _validate_frozen_numeric_artifact(
                        value,
                        path.relative_to(repo_root).as_posix(),
                        kind="tolerance table",
                    )
                )

    definition_root = repo_root / "benchmarks" / "definitions"
    if definition_root.is_dir():
        for path in sorted(definition_root.glob("*.yaml")):
            value = _load_yaml(path)
            issues.extend(
                _schema_issues(value, benchmark_definition_schema, path, repo_root)
            )
            if isinstance(value, dict):
                issues.extend(
                    _validate_frozen_numeric_artifact(
                        value,
                        path.relative_to(repo_root).as_posix(),
                        kind="benchmark definition",
                    )
                )

    baseline_root = repo_root / "benchmarks" / "baselines"
    if baseline_root.is_dir():
        for path in sorted(baseline_root.glob("**/*.json")):
            value = _load_json(path)
            issues.extend(
                _schema_issues(value, benchmark_baseline_schema, path, repo_root)
            )

    environment_root = repo_root / "environment"
    if environment_root.is_dir():
        for path in sorted(environment_root.glob("*.json")):
            if path.name.endswith(".local.json"):
                continue
            value = _load_json(path)
            issues.extend(
                _schema_issues(value, environment_lock_schema, path, repo_root)
            )

    return issues
