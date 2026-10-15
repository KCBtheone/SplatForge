"""Deterministic SHA-256 helpers used by specs, packets, and evidence."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


_BUFFER_SIZE = 1024 * 1024


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_json_digest(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def hash_file(path: str | os.PathLike[str]) -> str:
    file_path = Path(path)
    if file_path.is_symlink():
        raise ValueError(f"refusing to hash symlink: {file_path}")
    if not file_path.is_file():
        raise FileNotFoundError(file_path)

    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while chunk := handle.read(_BUFFER_SIZE):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def hash_path(path: str | os.PathLike[str]) -> str:
    """Hash a file or a directory without following symbolic links.

    Directory hashes include each POSIX-normalized relative path, byte size,
    and file digest. This makes renames, path-case changes, and empty/non-empty
    file changes observable. Empty directories are intentionally not semantic
    inputs.
    """

    root = Path(path)
    if root.is_symlink():
        raise ValueError(f"refusing to hash symlink: {root}")
    if root.is_file():
        return hash_file(root)
    if not root.is_dir():
        raise FileNotFoundError(root)

    records: list[dict[str, Any]] = []
    for candidate in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if candidate.is_symlink():
            raise ValueError(f"refusing to hash symlink: {candidate}")
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(root).as_posix()
        records.append(
            {
                "path": relative,
                "size": candidate.stat().st_size,
                "digest": hash_file(candidate),
            }
        )
    return canonical_json_digest(records)

