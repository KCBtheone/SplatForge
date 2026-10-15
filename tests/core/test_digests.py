from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from splatforge.core.digests import hash_file, hash_path


class DigestTests(unittest.TestCase):
    def test_tree_digest_is_path_sensitive_and_stable(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "a").mkdir()
            (root / "a" / "value.txt").write_text("same", encoding="utf-8")
            first = hash_path(root)
            second = hash_path(root)
            self.assertEqual(first, second)
            (root / "a" / "value.txt").rename(root / "renamed.txt")
            self.assertNotEqual(first, hash_path(root))

    def test_file_digest_uses_sha256_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "value.bin"
            path.write_bytes(b"abc")
            self.assertEqual(
                hash_file(path),
                "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )


if __name__ == "__main__":
    unittest.main()

