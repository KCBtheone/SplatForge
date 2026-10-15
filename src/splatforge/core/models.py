"""Immutable contracts and metadata objects used by the Phase 0 runtime."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch

from .digests import canonical_json_digest
from .errors import InvalidOpKeyError, SchemaValidationError


_OP_KEY_PATTERN = re.compile(
    r"^(?P<namespace>[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*)\."
    r"(?P<name>[a-z][a-z0-9_]*)@(?P<major>[1-9][0-9]*)$"
)
_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True, order=True)
class OpKey:
    namespace: str
    name: str
    major: int

    @classmethod
    def parse(cls, value: str | "OpKey") -> "OpKey":
        if isinstance(value, cls):
            return value
        match = _OP_KEY_PATTERN.fullmatch(value)
        if match is None:
            raise InvalidOpKeyError(
                f"invalid operator key: {value!r}",
                details={"expected": "namespace.name@positive_major"},
            )
        return cls(
            namespace=match.group("namespace"),
            name=match.group("name"),
            major=int(match.group("major")),
        )

    def __str__(self) -> str:
        return f"{self.namespace}.{self.name}@{self.major}"


class CallValidator(Protocol):
    def __call__(
        self,
        operands: tuple[Any, ...],
        attributes: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


class Runner(Protocol):
    def __call__(self, *operands: Any, **attributes: Any) -> Any: ...


class SupportPredicate(Protocol):
    def __call__(
        self,
        signature: "CallSignature",
        policy: Any,
    ) -> "SupportDecision": ...


@dataclass(frozen=True, slots=True)
class OpDefinition:
    op_key: OpKey
    spec_revision: str
    validator: CallValidator

    def __post_init__(self) -> None:
        if _SHA256_PATTERN.fullmatch(self.spec_revision) is None:
            raise ValueError("spec_revision must be a sha256:<64 lowercase hex> digest")

    def validate(
        self,
        operands: tuple[Any, ...],
        attributes: Mapping[str, Any],
    ) -> dict[str, Any]:
        try:
            normalized = dict(self.validator(operands, attributes))
        except SchemaValidationError:
            raise
        except Exception as exc:
            raise SchemaValidationError(
                f"call does not satisfy {self.op_key} schema: {exc}",
                details={"op": str(self.op_key)},
            ) from exc
        return normalized


@dataclass(frozen=True, slots=True)
class DeclaredEligibility:
    devices: tuple[str, ...]
    dtypes: tuple[str, ...]
    layouts: tuple[str, ...]
    backward: bool
    backward_may_copy_grad_output: bool
    jvp: bool
    determinism_guarantee: str
    host_synchronization: str
    cuda_graph: bool
    execution_modes: tuple[str, ...]
    constraints: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ImplementationManifest:
    schema_version: int
    implementation_id: str
    provider: str
    provider_api: int
    op_key: OpKey
    op_spec_revision: str
    implementation_version: str
    lowering_target: str
    static_variant_token: str
    priority: int
    eligibility: DeclaredEligibility
    build_fingerprint: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ImplementationManifest":
        eligibility = data["declared_eligibility"]
        execution = eligibility.get("execution", {})
        manifest = cls(
            schema_version=int(data["schema_version"]),
            implementation_id=str(data["implementation_id"]),
            provider=str(data["provider"]),
            provider_api=int(data["provider_api"]),
            op_key=OpKey.parse(str(data["op"])),
            op_spec_revision=str(data["op_spec_revision"]),
            implementation_version=str(data["implementation_version"]),
            lowering_target=str(data["lowering_target"]),
            static_variant_token=str(data["static_variant_token"]),
            priority=int(data.get("priority", 0)),
            eligibility=DeclaredEligibility(
                devices=tuple(str(item) for item in eligibility["devices"]),
                dtypes=tuple(str(item) for item in eligibility["dtypes"]),
                layouts=tuple(str(item) for item in eligibility["layouts"]),
                backward=bool(eligibility["backward"]),
                backward_may_copy_grad_output=bool(
                    eligibility.get("backward_may_copy_grad_output", False)
                ),
                jvp=bool(eligibility.get("jvp", False)),
                determinism_guarantee=str(
                    eligibility.get("determinism_guarantee", "none")
                ),
                host_synchronization=str(
                    eligibility.get("host_synchronization", "none")
                ),
                cuda_graph=bool(eligibility.get("cuda_graph", False)),
                execution_modes=tuple(
                    mode
                    for mode, status in sorted(execution.items())
                    if status == "supported"
                ),
                constraints=dict(eligibility.get("constraints", {})),
            ),
            build_fingerprint=str(data["build"]["fingerprint"]),
        )
        if manifest.schema_version != 1:
            raise ValueError("unsupported implementation manifest schema_version")
        if manifest.provider_api != 1:
            raise ValueError("unsupported provider_api")
        if _SHA256_PATTERN.fullmatch(manifest.op_spec_revision) is None:
            raise ValueError("op_spec_revision must be a sha256 digest")
        return manifest


@dataclass(frozen=True, slots=True)
class TensorMetadata:
    device: str
    device_type: str
    dtype: str
    layout: str
    shape: tuple[Any, ...]
    stride: tuple[Any, ...]
    storage_offset: int
    is_contiguous: bool
    last_dimension_contiguous: bool
    requires_grad: bool

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "TensorMetadata":
        stride = tuple(tensor.stride())
        return cls(
            device=str(tensor.device),
            device_type=tensor.device.type,
            dtype=str(tensor.dtype).removeprefix("torch."),
            layout=str(tensor.layout).removeprefix("torch."),
            shape=tuple(tensor.shape),
            stride=stride,
            storage_offset=int(tensor.storage_offset()),
            is_contiguous=tensor.is_contiguous(),
            last_dimension_contiguous=(tensor.ndim == 0 or stride[-1] == 1),
            requires_grad=bool(tensor.requires_grad),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "device_type": self.device_type,
            "dtype": self.dtype,
            "layout": self.layout,
            "shape": [str(value) for value in self.shape],
            "stride": [str(value) for value in self.stride],
            "storage_offset": self.storage_offset,
            "is_contiguous": self.is_contiguous,
            "last_dimension_contiguous": self.last_dimension_contiguous,
            "requires_grad": self.requires_grad,
        }


def _static_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise SchemaValidationError("static attributes must be finite")
        return value
    if isinstance(value, tuple):
        return [_static_value(item) for item in value]
    if isinstance(value, list):
        return [_static_value(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _static_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    raise SchemaValidationError(
        f"unsupported static attribute type: {type(value).__name__}"
    )


@dataclass(frozen=True, slots=True)
class CallSignature:
    op_key: OpKey
    spec_revision: str
    tensor_operands: tuple[TensorMetadata, ...]
    operand_kinds: tuple[str, ...]
    attributes: Mapping[str, Any]
    grad_enabled: bool
    inference_mode: bool
    autocast_enabled: bool
    deterministic_algorithms: bool

    @classmethod
    def from_call(
        cls,
        definition: OpDefinition,
        operands: Sequence[Any],
        attributes: Mapping[str, Any],
    ) -> "CallSignature":
        tensor_metadata: list[TensorMetadata] = []
        operand_kinds: list[str] = []
        for operand in operands:
            if isinstance(operand, torch.Tensor):
                tensor_metadata.append(TensorMetadata.from_tensor(operand))
                operand_kinds.append("tensor")
            else:
                operand_kinds.append(type(operand).__name__)
        return cls(
            op_key=definition.op_key,
            spec_revision=definition.spec_revision,
            tensor_operands=tuple(tensor_metadata),
            operand_kinds=tuple(operand_kinds),
            attributes=_static_value(attributes),
            grad_enabled=torch.is_grad_enabled(),
            inference_mode=torch.is_inference_mode_enabled(),
            autocast_enabled=torch.is_autocast_enabled(),
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        )

    @property
    def requires_backward(self) -> bool:
        return (
            self.grad_enabled
            and not self.inference_mode
            and any(item.requires_grad for item in self.tensor_operands)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": str(self.op_key),
            "spec_revision": self.spec_revision,
            "tensor_operands": [item.to_dict() for item in self.tensor_operands],
            "operand_kinds": list(self.operand_kinds),
            "attributes": self.attributes,
            "grad_enabled": self.grad_enabled,
            "inference_mode": self.inference_mode,
            "autocast_enabled": self.autocast_enabled,
            "deterministic_algorithms": self.deterministic_algorithms,
        }

    @property
    def digest(self) -> str:
        return canonical_json_digest(self.to_dict())


@dataclass(frozen=True, slots=True)
class SupportDecision:
    supported: bool
    reasons: tuple[str, ...] = ()

    @classmethod
    def yes(cls) -> "SupportDecision":
        return cls(True, ())

    @classmethod
    def no(cls, *reasons: str) -> "SupportDecision":
        return cls(False, tuple(reasons))


@dataclass(frozen=True, slots=True)
class RegisteredImplementation:
    manifest: ImplementationManifest
    runner: Runner
    support: SupportPredicate | None = None


@dataclass(frozen=True, slots=True)
class CapabilityRecord:
    implementation_id: str
    status: str
    authority: str
    environment_id: str
    release_tier: str = "best_effort"
    evidence_digest: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {
            "untested",
            "unsupported",
            "experimental",
            "conformant",
            "quarantined",
            "disabled",
        }:
            raise ValueError("invalid capability status")
        if self.release_tier not in {"tier1", "tier2", "best_effort"}:
            raise ValueError("invalid release tier")


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    implementation_id: str
    provider: str
    supported: bool
    reasons: tuple[str, ...]
    capability_status: str
    release_tier: str
    build_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "implementation_id": self.implementation_id,
            "provider": self.provider,
            "supported": self.supported,
            "reasons": list(self.reasons),
            "capability_status": self.capability_status,
            "release_tier": self.release_tier,
            "build_fingerprint": self.build_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class DispatchExplanation:
    op_key: str
    spec_revision: str
    signature_digest: str
    snapshot_token: str
    selected_implementation_id: str | None
    candidates: tuple[CandidateEvaluation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op_key,
            "spec_revision": self.spec_revision,
            "signature_digest": self.signature_digest,
            "snapshot_token": self.snapshot_token,
            "selected_implementation_id": self.selected_implementation_id,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }

