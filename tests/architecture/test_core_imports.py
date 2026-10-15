from __future__ import annotations

import ast
import unittest
from pathlib import Path


class CoreImportBoundaryTests(unittest.TestCase):
    def test_core_does_not_import_domain_or_provider_modules(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        core_root = repo_root / "src" / "splatforge" / "core"
        forbidden = (
            "splatforge.camera",
            "splatforge.spatial",
            "splatforge.sampling",
            "splatforge.matching",
            "splatforge.gaussian",
            "splatforge.splat",
            "splatforge.mesh",
            "splatforge.providers",
            "splatforge.integrations",
        )
        violations: list[str] = []
        for path in sorted(core_root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    if node.level:
                        relative_package = path.parent.relative_to(
                            repo_root / "src"
                        ).parts
                        keep = len(relative_package) - (node.level - 1)
                        prefix = relative_package[:keep]
                        if node.module:
                            names.append(".".join((*prefix, node.module)))
                        else:
                            names.extend(
                                ".".join((*prefix, alias.name))
                                for alias in node.names
                            )
                    elif node.module:
                        names.append(node.module)
                for name in names:
                    if name.startswith(forbidden):
                        violations.append(f"{path.name}:{node.lineno}: {name}")
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
