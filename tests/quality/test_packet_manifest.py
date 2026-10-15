from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import jsonschema

from tools.quality.packet_manifest import (
    build_packet_manifest,
    verify_packet_manifest,
    write_packet_manifest,
)


class PacketManifestTests(unittest.TestCase):
    def test_packet_roundtrip_and_tamper_detection(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as raw:
            packet = Path(raw)
            (packet / "tests_public").mkdir()
            work_order = packet / "WORK_ORDER.yaml"
            work_order.write_text("status: draft_not_executable\n", encoding="utf-8")
            payload = packet / "tests_public" / "fixture.txt"
            payload.write_text("frozen fixture", encoding="utf-8")
            manifest = build_packet_manifest(
                packet,
                packet_id="SF-TEST-001",
                work_order_path="WORK_ORDER.yaml",
                status="draft",
            )
            schema = json.loads(
                (repo_root / "tools/schemas/packet_manifest.schema.json").read_text(
                    encoding="utf-8"
                )
            )
            jsonschema.Draft202012Validator(
                schema,
                format_checker=jsonschema.FormatChecker(),
            ).validate(manifest)
            write_packet_manifest(packet, manifest)
            self.assertEqual(verify_packet_manifest(packet), [])
            payload.write_text("changed fixture", encoding="utf-8")
            self.assertIn(
                "file inventory, size, or digest mismatch",
                verify_packet_manifest(packet),
            )

    def test_frozen_work_order_pins_non_self_referential_payload(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            packet = Path(raw)
            (packet / "tests_public").mkdir()
            (packet / "tests_public" / "fixture.txt").write_text(
                "frozen fixture", encoding="utf-8"
            )
            work_order = packet / "WORK_ORDER.yaml"
            work_order.write_text("status: draft_not_executable\n", encoding="utf-8")
            draft = build_packet_manifest(
                packet,
                packet_id="SF-TEST-002",
                work_order_path="WORK_ORDER.yaml",
                status="draft",
            )
            work_order.write_text(
                "status: frozen\n"
                "packet_digest_algorithm: splatforge-packet-payload-v1\n"
                "packet_digest_excludes:\n"
                "  - WORK_ORDER.yaml\n"
                "  - PACK_MANIFEST.json\n"
                "packet_digest: "
                + draft["payload"]["digest"]
                + "\n",
                encoding="utf-8",
            )
            frozen = build_packet_manifest(
                packet,
                packet_id="SF-TEST-002",
                work_order_path="WORK_ORDER.yaml",
                status="frozen",
            )
            self.assertEqual(frozen["payload"]["digest"], draft["payload"]["digest"])
            write_packet_manifest(packet, frozen)
            self.assertEqual(verify_packet_manifest(packet), [])


if __name__ == "__main__":
    unittest.main()
