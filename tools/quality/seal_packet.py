"""CLI for creating or verifying a content-addressed implementation packet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jsonschema

from tools.quality.packet_manifest import (
    build_packet_manifest,
    verify_packet_manifest,
    write_packet_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("packet", type=Path)
    parser.add_argument("--packet-id")
    parser.add_argument("--work-order")
    parser.add_argument("--status", choices=("draft", "frozen"), default="draft")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()

    schema = json.loads(
        (args.repo.resolve() / "tools/schemas/packet_manifest.schema.json").read_text(
            encoding="utf-8"
        )
    )
    validator = jsonschema.Draft202012Validator(
        schema,
        format_checker=jsonschema.FormatChecker(),
    )

    if args.verify:
        manifest_path = args.packet.resolve() / "PACK_MANIFEST.json"
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
        errors = [
            error.message
            for error in sorted(
                validator.iter_errors(value),
                key=lambda item: list(item.absolute_path),
            )
        ]
        errors.extend(verify_packet_manifest(args.packet))
        print(json.dumps({"status": "passed" if not errors else "failed", "errors": errors}, indent=2))
        return 0 if not errors else 1
    if not args.packet_id or not args.work_order:
        parser.error("--packet-id and --work-order are required when creating a manifest")

    manifest = build_packet_manifest(
        args.packet,
        packet_id=args.packet_id,
        work_order_path=args.work_order,
        status=args.status,
    )
    validator.validate(manifest)
    destination = write_packet_manifest(args.packet, manifest)
    print(json.dumps({"status": "written", "path": str(destination), "packet_digest": manifest["aggregate"]["digest"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
