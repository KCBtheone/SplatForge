from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from splatforge.core.digests import hash_path


class WorkOrderCliTests(unittest.TestCase):
    def _git(self, repo: Path, *args: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return completed.stdout.strip()

    def _run_gate(
        self,
        source_root: Path,
        repo: Path,
        candidate_head: str,
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(source_root / "src"), str(source_root)]
        )
        return subprocess.run(
            [
                sys.executable,
                str(source_root / "tools/quality/check_work_order_scope.py"),
                "--repo",
                str(repo),
                "--work-order",
                "work_orders/SF-TEST-CLI.yaml",
                "--policy-ref",
                "main",
                "--candidate-head",
                candidate_head,
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

    def test_gate_reads_policy_ref_and_rejects_protected_diff(self) -> None:
        source_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw)
            self._git(repo, "init")
            self._git(repo, "symbolic-ref", "HEAD", "refs/heads/main")
            self._git(repo, "config", "user.name", "SplatForge Test")
            self._git(repo, "config", "user.email", "test@example.invalid")

            spec = repo / "specs" / "op"
            spec.mkdir(parents=True)
            (spec / "contract.txt").write_text("frozen contract", encoding="utf-8")
            self._git(repo, "add", ".")
            self._git(repo, "commit", "-m", "frozen contract")
            base_commit = self._git(repo, "rev-parse", "HEAD")
            spec_digest = hash_path(spec)

            self._git(repo, "checkout", "-b", "candidate")
            provider = repo / "src" / "splatforge" / "providers" / "triton" / "matching"
            provider.mkdir(parents=True)
            (provider / "kernel.py").write_text("VALUE = 1\n", encoding="utf-8")
            self._git(repo, "add", ".")
            self._git(repo, "commit", "-m", "provider only")
            good_candidate = self._git(repo, "rev-parse", "HEAD")

            self._git(repo, "checkout", "main")
            work_order_path = repo / "work_orders" / "SF-TEST-CLI.yaml"
            work_order_path.parent.mkdir()
            protected_path = repo / "tools" / "quality" / "protected_paths.yaml"
            protected_path.parent.mkdir(parents=True)
            protected_path.write_text(
                "schema_version: 1\n"
                "policy_id: test-protected-paths\n"
                "protected_paths:\n"
                "  - specs/**\n"
                "  - work_orders/**\n"
                "  - tools/quality/**\n",
                encoding="utf-8",
            )
            work_order = {
                "schema_version": 1,
                "task_id": "SF-TEST-CLI",
                "status": "frozen",
                "base_commit": base_commit,
                "packet_digest": "sha256:" + "1" * 64,
                "packet_digest_algorithm": "splatforge-packet-payload-v1",
                "packet_digest_excludes": [
                    "WORK_ORDER.yaml",
                    "PACK_MANIFEST.json",
                ],
                "op_id": "testing.scale_shift@1",
                "op_spec_revision": "sha256:" + "2" * 64,
                "implementation_id": "splatforge.triton.testing.scale_shift.v1",
                "goal": "Exercise the protected provider scope gate.",
                "network_mode": "off",
                "fresh_context_required": True,
                "allowed_paths": ["src/splatforge/providers/triton/matching/**"],
                "forbidden_paths": ["specs/**"],
                "frozen_inputs": {
                    "spec": {"path": "specs/op", "digest": spec_digest}
                },
                "required_commands": ["run tests"],
                "required_self_reports": ["report.json"],
                "automatic_failure": ["scope violation"],
                "stop_conditions": ["contract mismatch"],
            }
            work_order_path.write_text(
                yaml.safe_dump(work_order, sort_keys=False),
                encoding="utf-8",
            )
            self._git(repo, "add", ".")
            self._git(repo, "commit", "-m", "protected work order")

            passed = self._run_gate(source_root, repo, good_candidate)
            self.assertEqual(passed.returncode, 0, passed.stderr + passed.stdout)
            self.assertEqual(json.loads(passed.stdout)["status"], "passed")

            self._git(repo, "checkout", "candidate")
            (spec / "contract.txt").write_text("weakened contract", encoding="utf-8")
            self._git(repo, "add", ".")
            self._git(repo, "commit", "-m", "forbidden contract change")
            bad_candidate = self._git(repo, "rev-parse", "HEAD")
            failed = self._run_gate(source_root, repo, bad_candidate)
            self.assertEqual(failed.returncode, 1, failed.stderr + failed.stdout)
            codes = {item["code"] for item in json.loads(failed.stdout)["errors"]}
            self.assertIn("protected_path", codes)
            self.assertIn("frozen_input_digest_mismatch", codes)

            self._git(repo, "checkout", "-b", "symlink-candidate", base_commit)
            blob = subprocess.run(
                ["git", "-C", str(repo), "hash-object", "-w", "--stdin"],
                input="kernel.py",
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            ).stdout.strip()
            link_path = "src/splatforge/providers/triton/matching/link.py"
            self._git(
                repo,
                "update-index",
                "--add",
                "--cacheinfo",
                "120000",
                blob,
                link_path,
            )
            self._git(repo, "commit", "-m", "symlink in allowed path")
            symlink_candidate = self._git(repo, "rev-parse", "HEAD")
            symlink_result = self._run_gate(source_root, repo, symlink_candidate)
            self.assertEqual(
                symlink_result.returncode,
                1,
                symlink_result.stderr + symlink_result.stdout,
            )
            symlink_codes = {
                item["code"]
                for item in json.loads(symlink_result.stdout)["errors"]
            }
            self.assertIn("symlink_path", symlink_codes)


if __name__ == "__main__":
    unittest.main()
