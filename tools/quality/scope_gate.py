"""Pure helpers for enforcing a provider work order's changed-path scope."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from splatforge.core.digests import hash_path


@dataclass(frozen=True, slots=True)
class ScopeViolation:
    code: str
    path: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "path": self.path, "message": self.message}


def normalize_repo_path(value: str) -> str:
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if not normalized or path.is_absolute() or ":" in path.parts[0]:
        raise ValueError(f"path must be repository-relative: {value!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"path contains a forbidden component: {value!r}")
    return path.as_posix()


def path_matches(path: str, pattern: str) -> bool:
    normalized_path = normalize_repo_path(path)
    normalized_pattern = normalize_repo_path(pattern)
    if normalized_pattern.endswith("/**"):
        prefix = normalized_pattern[:-3].rstrip("/")
        return normalized_path == prefix or normalized_path.startswith(prefix + "/")
    return fnmatch.fnmatchcase(normalized_path, normalized_pattern)


def _contains_symlink(repo_root: Path, relative_path: str) -> bool:
    current = repo_root.resolve()
    for part in PurePosixPath(relative_path).parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def validate_changed_paths(
    changed_paths: list[str] | tuple[str, ...],
    work_order: dict[str, Any],
    repo_root: str | Path,
    *,
    protected_patterns: list[str] | tuple[str, ...] = (),
) -> list[ScopeViolation]:
    root = Path(repo_root)
    allowed = tuple(str(value) for value in work_order.get("allowed_paths", ()))
    forbidden = tuple(str(value) for value in work_order.get("forbidden_paths", ()))
    violations: list[ScopeViolation] = []

    for raw_path in changed_paths:
        try:
            path = normalize_repo_path(raw_path)
        except ValueError as exc:
            violations.append(ScopeViolation("invalid_path", raw_path, str(exc)))
            continue
        if any(path_matches(path, pattern) for pattern in protected_patterns):
            violations.append(
                ScopeViolation(
                    "protected_path",
                    path,
                    "path matches the policy loaded from the trusted ref",
                )
            )
            continue
        if any(path_matches(path, pattern) for pattern in forbidden):
            violations.append(
                ScopeViolation("forbidden_path", path, "path matches work-order forbidden_paths")
            )
            continue
        if not any(path_matches(path, pattern) for pattern in allowed):
            violations.append(
                ScopeViolation("outside_allowlist", path, "path is not covered by work-order allowed_paths")
            )
            continue
        if _contains_symlink(root, path):
            violations.append(
                ScopeViolation("symlink_path", path, "allowed path traverses a symbolic link")
            )
    return violations


def validate_frozen_inputs(
    work_order: dict[str, Any],
    repo_root: str | Path,
) -> list[ScopeViolation]:
    root = Path(repo_root).resolve()
    violations: list[ScopeViolation] = []
    for name, item in work_order.get("frozen_inputs", {}).items():
        raw_path = str(item.get("path", ""))
        try:
            path = normalize_repo_path(raw_path.rstrip("/"))
            if _contains_symlink(root, path):
                raise ValueError("frozen input traverses a symbolic link")
            target = root / Path(path)
            actual = hash_path(target)
        except (ValueError, OSError) as exc:
            violations.append(
                ScopeViolation("frozen_input_unreadable", raw_path, f"{name}: {exc}")
            )
            continue
        expected = str(item.get("digest", ""))
        if actual != expected:
            violations.append(
                ScopeViolation(
                    "frozen_input_digest_mismatch",
                    path,
                    f"{name}: expected {expected}, observed {actual}",
                )
            )
    return violations
