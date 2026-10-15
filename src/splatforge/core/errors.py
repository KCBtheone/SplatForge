"""Structured errors exposed by the SplatForge runtime."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class SplatForgeError(RuntimeError):
    """Base error with a stable machine-readable code and details."""

    code = "splatforge_error"

    def __init__(
        self,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.details = dict(details or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": str(self),
            "details": self.details,
        }


class InvalidOpKeyError(SplatForgeError):
    code = "invalid_op_key"


class DefinitionConflictError(SplatForgeError):
    code = "definition_conflict"


class ImplementationConflictError(SplatForgeError):
    code = "implementation_conflict"


class UnknownOperatorError(SplatForgeError):
    code = "unknown_operator"


class SpecRevisionMismatchError(SplatForgeError):
    code = "spec_revision_mismatch"


class SchemaValidationError(SplatForgeError):
    code = "schema_validation_failed"


class DispatchError(SplatForgeError):
    code = "dispatch_failed"


class RuntimeSealedError(SplatForgeError):
    code = "runtime_sealed"


class PlanSignatureMismatchError(SplatForgeError):
    code = "plan_signature_mismatch"


class ManifestValidationError(SplatForgeError):
    code = "manifest_validation_failed"


class DigestMismatchError(SplatForgeError):
    code = "digest_mismatch"


class PathScopeViolationError(SplatForgeError):
    code = "path_scope_violation"

