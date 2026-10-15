"""CLI: validate every protected manifest and its cross-file digests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.quality.manifest_validation import validate_repository


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    issues = validate_repository(args.repo)
    payload = {
        "status": "passed" if not issues else "failed",
        "issue_count": len(issues),
        "issues": [issue.to_dict() for issue in issues],
    }
    if args.as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"manifest validation: {payload['status']} ({len(issues)} issue(s))")
        for issue in issues:
            print(f"{issue.path}: {issue.message}")
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())

