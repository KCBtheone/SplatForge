"""CLI gate that reads a work order from a protected policy ref."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import jsonschema
import yaml

from splatforge.core.digests import sha256_bytes
from tools.quality.manifest_validation import (
    find_placeholders,
    validate_work_order_semantics,
)
from tools.quality.scope_gate import (
    normalize_repo_path,
    validate_changed_paths,
    validate_frozen_inputs,
)


def _git(repo: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.decode("utf-8", errors="replace").strip())
    return completed.stdout


def _tree_entry(repo: Path, commit: str, path: str) -> tuple[str, str] | None:
    output = _git(repo, "ls-tree", "-z", commit, "--", path)
    if not output:
        return None
    header, separator, _listed_path = output.partition(b"\t")
    if not separator:
        raise RuntimeError(f"could not parse git tree entry for {path}")
    fields = header.decode("ascii").split()
    if len(fields) != 3:
        raise RuntimeError(f"could not parse git tree metadata for {path}")
    mode, object_type, _object_id = fields
    return mode, object_type


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--work-order", required=True)
    parser.add_argument(
        "--policy-ref",
        required=True,
        help="trusted ref containing the protected work order and gate policy",
    )
    parser.add_argument(
        "--candidate-head",
        default="HEAD",
        help="candidate commit whose provider-only diff is checked",
    )
    parser.add_argument("--expected-work-order-digest")
    args = parser.parse_args()

    repo = args.repo.resolve()
    try:
        work_order_path = normalize_repo_path(args.work_order)
        policy_commit = _git(
            repo, "rev-parse", "--verify", f"{args.policy_ref}^{{commit}}"
        ).decode("ascii").strip()
        candidate_commit = _git(
            repo, "rev-parse", "--verify", f"{args.candidate_head}^{{commit}}"
        ).decode("ascii").strip()
        base_bytes = _git(repo, "show", f"{policy_commit}:{work_order_path}")
        protected_bytes = _git(
            repo,
            "show",
            f"{policy_commit}:tools/quality/protected_paths.yaml",
        )
        protected_policy = yaml.safe_load(protected_bytes.decode("utf-8"))
        if not isinstance(protected_policy, dict):
            raise RuntimeError("protected-path policy root is not a mapping")
        if protected_policy.get("schema_version") != 1:
            raise RuntimeError("unsupported protected-path policy schema")
        protected_patterns = tuple(
            str(value) for value in protected_policy.get("protected_paths", ())
        )
        if not protected_patterns or len(set(protected_patterns)) != len(
            protected_patterns
        ):
            raise RuntimeError("protected-path policy must contain unique patterns")
        for pattern in protected_patterns:
            normalize_repo_path(pattern)
        base_order = yaml.safe_load(base_bytes.decode("utf-8"))
        if not isinstance(base_order, dict):
            raise RuntimeError("protected work order root is not a mapping")
        base_commit = str(base_order.get("base_commit", ""))
        _git(repo, "merge-base", "--is-ancestor", base_commit, candidate_commit)
        changed_raw = _git(
            repo,
            "diff",
            "--name-only",
            "-z",
            f"{base_commit}...{candidate_commit}",
        )
        changed = [
            item.decode("utf-8")
            for item in changed_raw.split(b"\0")
            if item
        ]
    except (OSError, RuntimeError, UnicodeError) as exc:
        payload = {"status": "failed", "errors": [{"code": "git_preflight", "message": str(exc)}]}
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    errors: list[dict[str, str]] = []
    schema_path = Path(__file__).resolve().parents[1] / "schemas/work_order.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    for error in sorted(validator.iter_errors(base_order), key=lambda item: list(item.path)):
        location = ".".join(str(item) for item in error.absolute_path)
        errors.append(
            {
                "code": "work_order_schema_invalid",
                "message": f"{error.message} ({location or '$'})",
            }
        )
    if base_order.get("status") != "frozen":
        errors.append({"code": "work_order_not_frozen", "message": "implementation gate requires status=frozen"})
    for issue in validate_work_order_semantics(base_order, args.work_order):
        errors.append({"code": "work_order_invalid", "message": issue.message})
    placeholders = find_placeholders(base_order)
    if placeholders:
        errors.append({"code": "work_order_placeholder", "message": ", ".join(placeholders)})

    work_order_digest = sha256_bytes(base_bytes)
    if (
        args.expected_work_order_digest is not None
        and args.expected_work_order_digest != work_order_digest
    ):
        errors.append({"code": "work_order_digest_mismatch", "message": f"expected {args.expected_work_order_digest}, observed {work_order_digest}"})

    errors.extend(
        item.to_dict()
        for item in validate_changed_paths(
            changed,
            base_order,
            repo,
            protected_patterns=protected_patterns,
        )
    )
    for path in changed:
        try:
            entry = _tree_entry(repo, candidate_commit, path)
        except RuntimeError as exc:
            errors.append(
                {
                    "code": "git_tree_inspection_failed",
                    "path": path,
                    "message": str(exc),
                }
            )
            continue
        if entry is None:
            continue
        mode, object_type = entry
        if mode == "120000":
            errors.append(
                {
                    "code": "symlink_path",
                    "path": path,
                    "message": "candidate tree contains a symbolic link",
                }
            )
        if mode == "160000" or object_type == "commit":
            errors.append(
                {
                    "code": "submodule_path",
                    "path": path,
                    "message": "candidate tree contains a submodule entry",
                }
            )
    errors.extend(item.to_dict() for item in validate_frozen_inputs(base_order, repo))

    payload = {
        "status": "passed" if not errors else "failed",
        "work_order_digest": work_order_digest,
        "base_commit": base_commit,
        "policy_ref": policy_commit,
        "candidate_head": candidate_commit,
        "changed_paths": changed,
        "errors": errors,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
