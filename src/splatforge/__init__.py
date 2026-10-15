"""Public Phase 0 runtime surface for SplatForge."""

from .core.errors import SplatForgeError
from .core.models import (
    CapabilityRecord,
    DispatchExplanation,
    ImplementationManifest,
    OpDefinition,
    OpKey,
)
from .core.policy import ExecutionPolicy
from .core.runtime import ExecutionPlan, Runtime

__all__ = [
    "CapabilityRecord",
    "DispatchExplanation",
    "ExecutionPlan",
    "ExecutionPolicy",
    "ImplementationManifest",
    "OpDefinition",
    "OpKey",
    "Runtime",
    "SplatForgeError",
]

__version__ = "0.0.1a1"

