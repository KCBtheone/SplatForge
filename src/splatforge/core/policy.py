"""Provider-selection policy, kept separate from mathematical semantics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExecutionPolicy:
    providers: tuple[str, ...] = ("native_cuda", "triton", "torch")
    implementation_id: str | None = None
    fixed_provider: str | None = None
    runtime_failure: str = "error"
    copies: str = "forbid"
    host_sync: str = "forbid"
    precision: str = "balanced"
    determinism: str = "inherit"
    autocast: str = "inherit"
    evidence_requirement: str = "release_verified"
    allow_unverified: bool = False
    require_cuda_graph: bool = False

    def __post_init__(self) -> None:
        if not self.providers:
            raise ValueError("providers cannot be empty")
        if len(set(self.providers)) != len(self.providers):
            raise ValueError("providers cannot contain duplicates")
        if self.implementation_id is not None and self.fixed_provider is not None:
            raise ValueError("implementation_id and fixed_provider are mutually exclusive")
        if self.runtime_failure != "error":
            raise ValueError("Phase 0 permits runtime_failure='error' only")
        if self.copies not in {"forbid", "allow_declared"}:
            raise ValueError("copies must be 'forbid' or 'allow_declared'")
        if self.host_sync not in {"forbid", "allow_declared"}:
            raise ValueError("host_sync must be 'forbid' or 'allow_declared'")
        if self.determinism not in {"inherit", "require", "allow"}:
            raise ValueError("invalid determinism policy")
        if self.autocast not in {"inherit", "forbid", "explicit"}:
            raise ValueError("invalid autocast policy")
        if self.evidence_requirement not in {
            "any",
            "conformant",
            "release_verified",
        }:
            raise ValueError("invalid evidence_requirement")
        if self.allow_unverified and self.implementation_id is None:
            raise ValueError(
                "allow_unverified requires an exact implementation_id; "
                "auto/provider-wide unverified selection is forbidden"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "providers": list(self.providers),
            "implementation_id": self.implementation_id,
            "fixed_provider": self.fixed_provider,
            "runtime_failure": self.runtime_failure,
            "copies": self.copies,
            "host_sync": self.host_sync,
            "precision": self.precision,
            "determinism": self.determinism,
            "autocast": self.autocast,
            "evidence_requirement": self.evidence_requirement,
            "allow_unverified": self.allow_unverified,
            "require_cuda_graph": self.require_cuda_graph,
        }

