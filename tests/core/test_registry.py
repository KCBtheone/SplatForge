from __future__ import annotations

import unittest

from splatforge.core.errors import DefinitionConflictError, RuntimeSealedError
from splatforge.core.models import OpDefinition, OpKey
from splatforge.core.registry import OpDefinitionRegistry


def _validator(_operands, attributes):
    return attributes


class DefinitionRegistryTests(unittest.TestCase):
    def test_registration_is_idempotent_for_same_revision(self) -> None:
        registry = OpDefinitionRegistry()
        definition = OpDefinition(
            OpKey.parse("testing.identity@1"),
            "sha256:" + "a" * 64,
            _validator,
        )
        self.assertTrue(registry.register(definition))
        self.assertFalse(registry.register(definition))
        self.assertEqual(registry.generation, 1)

    def test_conflicting_batch_is_transactional(self) -> None:
        registry = OpDefinitionRegistry()
        first = OpDefinition(
            OpKey.parse("testing.identity@1"),
            "sha256:" + "a" * 64,
            _validator,
        )
        conflict = OpDefinition(
            OpKey.parse("testing.identity@1"),
            "sha256:" + "b" * 64,
            _validator,
        )
        with self.assertRaises(DefinitionConflictError):
            registry.register_many((first, conflict))
        self.assertEqual(registry.generation, 0)
        self.assertEqual(registry.items(), ())

    def test_seal_blocks_late_registration(self) -> None:
        registry = OpDefinitionRegistry()
        registry.seal()
        with self.assertRaises(RuntimeSealedError):
            registry.register(
                OpDefinition(
                    OpKey.parse("testing.identity@1"),
                    "sha256:" + "a" * 64,
                    _validator,
                )
            )


if __name__ == "__main__":
    unittest.main()

