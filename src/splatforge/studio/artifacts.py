"""Atomic run records and checkpoint helpers."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, device="cpu"):
    # Only load this application's user-owned training checkpoints. CLI does
    # not load arbitrary remote pickle files or silently download checkpoints.
    return torch.load(path, map_location=device, weights_only=False)


class RunLock:
    """OS-released exclusive lock; a crash cannot leave a stale lock holder."""

    def __init__(self, output):
        self.path = Path(output) / "run.lock"
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+b")
        if self.path.stat().st_size == 0:
            self.handle.write(b"0")
            self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.handle.close()
            self.handle = None
            raise RuntimeError(f"another trainer owns {self.path.parent}") from exc
        return self

    def __exit__(self, *_):
        if self.handle:
            if os.name == "nt":
                import msvcrt

                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            self.handle.close()


class StorageManifest:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.path = self.root / "storage_manifest.json"
        self.items = {}
        if self.path.is_file():
            self.items = {
                row["path"]: row for row in json.loads(self.path.read_text())["items"]
            }

    def record(
        self,
        path,
        decision,
        purpose,
        *,
        next_use=None,
        release_after=None,
        state="IN_USE",
    ):
        resolved = Path(path).resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError("artifact must belong to this run")
        self.items[str(resolved)] = {
            "path": str(resolved),
            "owner": str(self.root),
            "decision": decision,
            "purpose": purpose,
            "next_use": next_use,
            "release_after": release_after,
            "state": state,
            "updated": timestamp(),
        }
        atomic_json(
            self.path, {"run_id": self.root.name, "items": list(self.items.values())}
        )

    def reclaim_checkpoint(self, path):
        resolved = Path(path).resolve()
        record = self.items.get(str(resolved))
        if (
            not record
            or record["decision"] != "RECLAIM"
            or record["state"] != "READY"
            or not resolved.is_relative_to(self.root)
        ):
            raise RuntimeError("checkpoint is not registered and ready for reclamation")
        if resolved.is_file() and not resolved.is_symlink():
            resolved.unlink()
        record["state"] = "DELETED"
        record["updated"] = timestamp()
        atomic_json(
            self.path, {"run_id": self.root.name, "items": list(self.items.values())}
        )
