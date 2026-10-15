"""Content-address and independently verify an already assembled packet."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from splatforge.core.digests import canonical_json_digest, hash_file
from tools.quality.manifest_validation import find_placeholders
from tools.quality.scope_gate import normalize_repo_path


MANIFEST_NAME = "PACK_MANIFEST.json"
ALGORITHM = "splatforge-packet-v1"
PAYLOAD_ALGORITHM = "splatforge-packet-payload-v1"


def _entries(packet_root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(packet_root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ValueError(f"packet cannot contain symbolic links: {path}")
        if not path.is_file() or path.name == MANIFEST_NAME:
            continue
        relative = normalize_repo_path(path.relative_to(packet_root).as_posix())
        entries.append(
            {"path": relative, "size": path.stat().st_size, "digest": hash_file(path)}
        )
    return entries


def _aggregate_payload(
    packet_id: str,
    status: str,
    work_order: dict[str, str],
    files: list[dict[str, Any]],
    payload_digest: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "packet_id": packet_id,
        "status": status,
        "work_order": work_order,
        "files": files,
        "payload": {
            "algorithm": PAYLOAD_ALGORITHM,
            "digest": payload_digest,
        },
        "algorithm": ALGORITHM,
    }


def _packet_payload_digest(
    files: list[dict[str, Any]],
    work_order_path: str,
) -> str:
    payload_files = [item for item in files if item["path"] != work_order_path]
    return canonical_json_digest(
        {
            "schema_version": 1,
            "algorithm": PAYLOAD_ALGORITHM,
            "files": payload_files,
        }
    )


def build_packet_manifest(
    packet_root: str | Path,
    *,
    packet_id: str,
    work_order_path: str,
    status: str = "draft",
) -> dict[str, Any]:
    root = Path(packet_root).resolve()
    if status not in {"draft", "frozen"}:
        raise ValueError("packet status must be draft or frozen")
    normalized_work_order = normalize_repo_path(work_order_path)
    work_order_file = root / Path(normalized_work_order)
    if not work_order_file.is_file() or work_order_file.is_symlink():
        raise FileNotFoundError(work_order_file)
    work_order = {
        "path": normalized_work_order,
        "digest": hash_file(work_order_file),
    }
    files = _entries(root)
    payload_digest = _packet_payload_digest(files, normalized_work_order)
    if status == "frozen":
        work_order_value = yaml.safe_load(work_order_file.read_text(encoding="utf-8"))
        if not isinstance(work_order_value, dict):
            raise ValueError("frozen work order root must be a mapping")
        if work_order_value.get("status") != "frozen":
            raise ValueError("a frozen packet requires a frozen work order")
        if work_order_value.get("packet_digest_algorithm") != PAYLOAD_ALGORITHM:
            raise ValueError("frozen work order declares the wrong packet digest algorithm")
        expected_excludes = {normalized_work_order, MANIFEST_NAME}
        if set(work_order_value.get("packet_digest_excludes", ())) != expected_excludes:
            raise ValueError(
                "frozen work order must exclude its own path and PACK_MANIFEST.json"
            )
        placeholders = find_placeholders(work_order_value)
        if placeholders:
            raise ValueError(
                "frozen work order contains placeholders at " + ", ".join(placeholders)
            )
        if work_order_value.get("packet_digest") != payload_digest:
            raise ValueError(
                "work-order packet_digest must equal the packet payload digest "
                "(excluding WORK_ORDER and PACK_MANIFEST)"
            )
    aggregate = canonical_json_digest(
        _aggregate_payload(packet_id, status, work_order, files, payload_digest)
    )
    return {
        "schema_version": 1,
        "packet_id": packet_id,
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "work_order": work_order,
        "files": files,
        "payload": {
            "algorithm": PAYLOAD_ALGORITHM,
            "digest": payload_digest,
        },
        "aggregate": {"algorithm": ALGORITHM, "digest": aggregate},
    }


def write_packet_manifest(packet_root: str | Path, manifest: dict[str, Any]) -> Path:
    destination = Path(packet_root).resolve() / MANIFEST_NAME
    destination.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def verify_packet_manifest(packet_root: str | Path) -> list[str]:
    root = Path(packet_root).resolve()
    manifest_path = root / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []

    observed_files = _entries(root)
    if observed_files != manifest.get("files"):
        errors.append("file inventory, size, or digest mismatch")
    work_order = manifest.get("work_order", {})
    payload = manifest.get("payload", {})
    work_order_file: Path | None = None
    try:
        work_order_file = root / Path(normalize_repo_path(str(work_order["path"])))
        observed_work_order_digest = hash_file(work_order_file)
        if observed_work_order_digest != work_order.get("digest"):
            errors.append("work-order digest mismatch")
    except (KeyError, OSError, ValueError) as exc:
        errors.append(f"work order is not verifiable: {exc}")

    try:
        observed_payload_digest = _packet_payload_digest(
            observed_files,
            str(work_order["path"]),
        )
        if payload.get("algorithm") != PAYLOAD_ALGORITHM:
            errors.append("unknown payload digest algorithm")
        if payload.get("digest") != observed_payload_digest:
            errors.append("packet payload digest mismatch")
        if work_order_file is None:
            raise ValueError("work-order path is unavailable")
        work_order_value = yaml.safe_load(work_order_file.read_text(encoding="utf-8"))
        if manifest.get("status") == "frozen":
            if not isinstance(work_order_value, dict) or work_order_value.get("status") != "frozen":
                errors.append("frozen packet does not contain a frozen work order")
            elif work_order_value.get("packet_digest_algorithm") != PAYLOAD_ALGORITHM:
                errors.append("frozen work order declares the wrong payload algorithm")
            elif set(work_order_value.get("packet_digest_excludes", ())) != {
                str(work_order["path"]),
                MANIFEST_NAME,
            }:
                errors.append("frozen work order declares the wrong payload exclusions")
            elif work_order_value.get("packet_digest") != observed_payload_digest:
                errors.append("frozen work order does not pin the packet payload digest")
    except (KeyError, OSError, UnicodeError, ValueError, yaml.YAMLError) as exc:
        observed_payload_digest = str(payload.get("digest", ""))
        errors.append(f"packet payload is not verifiable: {exc}")

    payload = _aggregate_payload(
        str(manifest.get("packet_id", "")),
        str(manifest.get("status", "")),
        dict(work_order),
        list(manifest.get("files", [])),
        observed_payload_digest,
    )
    observed_aggregate = canonical_json_digest(payload)
    if manifest.get("aggregate", {}).get("algorithm") != ALGORITHM:
        errors.append("unknown aggregate algorithm")
    if manifest.get("aggregate", {}).get("digest") != observed_aggregate:
        errors.append("aggregate packet digest mismatch")
    return errors
