"""Non-overwritable, independently instantiable runtime registries."""

from __future__ import annotations

from collections.abc import Iterable

from .errors import (
    DefinitionConflictError,
    ImplementationConflictError,
    RuntimeSealedError,
    UnknownOperatorError,
)
from .models import OpDefinition, OpKey, RegisteredImplementation


class OpDefinitionRegistry:
    def __init__(self) -> None:
        self._items: dict[OpKey, OpDefinition] = {}
        self._generation = 0
        self._sealed = False

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def sealed(self) -> bool:
        return self._sealed

    def seal(self) -> None:
        self._sealed = True

    def register(self, definition: OpDefinition) -> bool:
        return self.register_many((definition,)) > 0

    def register_many(self, definitions: Iterable[OpDefinition]) -> int:
        if self._sealed:
            raise RuntimeSealedError("definition registry is sealed")
        pending = tuple(definitions)
        seen: dict[OpKey, OpDefinition] = {}
        for definition in pending:
            current = seen.get(definition.op_key) or self._items.get(definition.op_key)
            if current is None:
                seen[definition.op_key] = definition
                continue
            if current.spec_revision != definition.spec_revision:
                raise DefinitionConflictError(
                    f"{definition.op_key} already has a different spec revision",
                    details={
                        "registered": current.spec_revision,
                        "requested": definition.spec_revision,
                    },
                )

        additions = 0
        for definition in pending:
            if definition.op_key not in self._items:
                self._items[definition.op_key] = definition
                additions += 1
        if additions:
            self._generation += 1
        return additions

    def get(self, op_key: str | OpKey) -> OpDefinition:
        key = OpKey.parse(op_key)
        try:
            return self._items[key]
        except KeyError as exc:
            raise UnknownOperatorError(
                f"operator is not registered: {key}",
                details={"op": str(key)},
            ) from exc

    def items(self) -> tuple[OpDefinition, ...]:
        return tuple(self._items[key] for key in sorted(self._items))


class ImplementationRegistry:
    def __init__(self) -> None:
        self._by_id: dict[str, RegisteredImplementation] = {}
        self._by_op: dict[OpKey, list[str]] = {}
        self._generation = 0
        self._sealed = False

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def sealed(self) -> bool:
        return self._sealed

    def seal(self) -> None:
        self._sealed = True

    def register(self, implementation: RegisteredImplementation) -> None:
        self.register_many((implementation,))

    def register_many(self, implementations: Iterable[RegisteredImplementation]) -> None:
        if self._sealed:
            raise RuntimeSealedError("implementation registry is sealed")
        pending = tuple(implementations)
        seen_ids = set(self._by_id)
        for implementation in pending:
            identifier = implementation.manifest.implementation_id
            if identifier in seen_ids:
                raise ImplementationConflictError(
                    f"implementation ID is already registered: {identifier}",
                    details={"implementation_id": identifier},
                )
            seen_ids.add(identifier)

        for implementation in pending:
            manifest = implementation.manifest
            self._by_id[manifest.implementation_id] = implementation
            self._by_op.setdefault(manifest.op_key, []).append(
                manifest.implementation_id
            )
        if pending:
            self._generation += 1

    def for_operator(self, op_key: str | OpKey) -> tuple[RegisteredImplementation, ...]:
        key = OpKey.parse(op_key)
        return tuple(self._by_id[item] for item in self._by_op.get(key, ()))

    def get(self, implementation_id: str) -> RegisteredImplementation | None:
        return self._by_id.get(implementation_id)

    def items(self) -> tuple[RegisteredImplementation, ...]:
        return tuple(self._by_id[key] for key in sorted(self._by_id))

