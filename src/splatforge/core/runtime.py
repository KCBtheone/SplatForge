"""Metadata-only planning and exact-implementation execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .digests import canonical_json_digest
from .errors import (
    DispatchError,
    PlanSignatureMismatchError,
    RuntimeSealedError,
    SpecRevisionMismatchError,
)
from .models import (
    CallSignature,
    CandidateEvaluation,
    CapabilityRecord,
    DispatchExplanation,
    ImplementationManifest,
    OpDefinition,
    RegisteredImplementation,
    Runner,
    SupportDecision,
    SupportPredicate,
)
from .policy import ExecutionPolicy
from .registry import ImplementationRegistry, OpDefinitionRegistry


@dataclass(frozen=True, slots=True)
class RuntimeSnapshot:
    definition_generation: int
    implementation_generation: int
    capability_generation: int
    token: str


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    runtime: "Runtime"
    implementation: RegisteredImplementation
    definition: OpDefinition
    signature: CallSignature
    attributes: dict[str, Any]
    policy: ExecutionPolicy
    snapshot_token: str

    @property
    def implementation_id(self) -> str:
        return self.implementation.manifest.implementation_id

    def run(self, *operands: Any, **attributes: Any) -> Any:
        normalized = self.definition.validate(tuple(operands), attributes)
        current = CallSignature.from_call(self.definition, operands, normalized)
        if current.digest != self.signature.digest:
            raise PlanSignatureMismatchError(
                "execution inputs do not match the metadata used to build the plan",
                details={
                    "planned_signature": self.signature.digest,
                    "received_signature": current.digest,
                    "implementation_id": self.implementation_id,
                },
            )
        if normalized != self.attributes:
            raise PlanSignatureMismatchError(
                "normalized static attributes differ from the planned call",
                details={"implementation_id": self.implementation_id},
            )
        # Runtime failure is deliberately terminal. There is no retry/fallback.
        return self.implementation.runner(*operands, **normalized)


class Runtime:
    def __init__(self) -> None:
        self.definitions = OpDefinitionRegistry()
        self.implementations = ImplementationRegistry()
        self._capabilities: dict[str, CapabilityRecord] = {}
        self._capability_generation = 0

    def register_definition(self, definition: OpDefinition) -> None:
        self.definitions.register(definition)

    def register_implementation(
        self,
        manifest: ImplementationManifest,
        runner: Runner,
        *,
        support: SupportPredicate | None = None,
    ) -> None:
        definition = self.definitions.get(manifest.op_key)
        if definition.spec_revision != manifest.op_spec_revision:
            raise SpecRevisionMismatchError(
                f"implementation {manifest.implementation_id} targets the wrong spec",
                details={
                    "operator": str(manifest.op_key),
                    "definition_revision": definition.spec_revision,
                    "implementation_revision": manifest.op_spec_revision,
                },
            )
        self.implementations.register(
            RegisteredImplementation(manifest=manifest, runner=runner, support=support)
        )

    def record_capability(self, record: CapabilityRecord) -> None:
        if self.definitions.sealed or self.implementations.sealed:
            raise RuntimeSealedError(
                "capability catalog cannot change after runtime.seal()"
            )
        if self.implementations.get(record.implementation_id) is None:
            raise ValueError(
                f"capability references unknown implementation: {record.implementation_id}"
            )
        if self._capabilities.get(record.implementation_id) == record:
            return
        self._capabilities[record.implementation_id] = record
        self._capability_generation += 1

    def snapshot(self) -> RuntimeSnapshot:
        payload = {
            "definition_generation": self.definitions.generation,
            "definitions": [
                {"op": str(item.op_key), "spec_revision": item.spec_revision}
                for item in self.definitions.items()
            ],
            "implementation_generation": self.implementations.generation,
            "implementations": [
                {
                    "implementation_id": item.manifest.implementation_id,
                    "op": str(item.manifest.op_key),
                    "spec_revision": item.manifest.op_spec_revision,
                    "build_fingerprint": item.manifest.build_fingerprint,
                }
                for item in self.implementations.items()
            ],
            "capability_generation": self._capability_generation,
            "capabilities": [
                {
                    "implementation_id": item.implementation_id,
                    "status": item.status,
                    "environment_id": item.environment_id,
                    "release_tier": item.release_tier,
                    "evidence_digest": item.evidence_digest,
                }
                for item in sorted(
                    self._capabilities.values(), key=lambda value: value.implementation_id
                )
            ],
        }
        return RuntimeSnapshot(
            definition_generation=self.definitions.generation,
            implementation_generation=self.implementations.generation,
            capability_generation=self._capability_generation,
            token=canonical_json_digest(payload),
        )

    def seal(self) -> RuntimeSnapshot:
        self.definitions.seal()
        self.implementations.seal()
        return self.snapshot()

    def _prepare_call(
        self,
        op: str,
        operands: tuple[Any, ...],
        attributes: dict[str, Any],
    ) -> tuple[OpDefinition, dict[str, Any], CallSignature]:
        definition = self.definitions.get(op)
        normalized = definition.validate(operands, attributes)
        signature = CallSignature.from_call(definition, operands, normalized)
        return definition, normalized, signature

    @staticmethod
    def _layout_names(signature: CallSignature) -> list[str]:
        result: list[str] = []
        for metadata in signature.tensor_operands:
            if metadata.is_contiguous:
                result.append("contiguous")
            elif metadata.last_dimension_contiguous:
                result.append("strided_last_dim")
            else:
                result.append("strided")
        return result

    def _declared_support(
        self,
        manifest: ImplementationManifest,
        signature: CallSignature,
        policy: ExecutionPolicy,
    ) -> SupportDecision:
        reasons: list[str] = []
        eligibility = manifest.eligibility

        if any(
            operand.device_type not in eligibility.devices
            for operand in signature.tensor_operands
        ):
            reasons.append("unsupported_device")
        if any(
            operand.dtype not in eligibility.dtypes
            for operand in signature.tensor_operands
        ):
            reasons.append("unsupported_dtype")
        if any(
            layout not in eligibility.layouts
            for layout in self._layout_names(signature)
        ):
            reasons.append("unsupported_layout")
        if signature.requires_backward and not eligibility.backward:
            reasons.append("no_backward")
        if (
            signature.requires_backward
            and eligibility.backward_may_copy_grad_output
            and policy.copies == "forbid"
        ):
            reasons.append("backward_copy_forbidden")
        if (
            eligibility.host_synchronization != "none"
            and policy.host_sync == "forbid"
        ):
            reasons.append("host_synchronization_forbidden")
        if policy.require_cuda_graph and not eligibility.cuda_graph:
            reasons.append("not_graph_safe")
        if (
            (policy.determinism == "require" or signature.deterministic_algorithms)
            and eligibility.determinism_guarantee == "none"
        ):
            reasons.append("determinism_requirement_not_met")

        capability = self._capabilities.get(manifest.implementation_id)
        if capability is None:
            capability_status = "untested"
            release_tier = "best_effort"
        else:
            capability_status = capability.status
            release_tier = capability.release_tier

        if capability_status in {"quarantined", "disabled", "unsupported"}:
            reasons.append(f"capability_{capability_status}")
        elif policy.allow_unverified:
            pass
        elif policy.evidence_requirement == "conformant":
            if capability_status != "conformant":
                reasons.append("conformance_evidence_required")
        elif policy.evidence_requirement == "release_verified":
            if capability_status != "conformant" or release_tier not in {
                "tier1",
                "tier2",
            }:
                reasons.append("release_evidence_required")

        if reasons:
            return SupportDecision.no(*reasons)
        return SupportDecision.yes()

    def explain_dispatch(
        self,
        op: str,
        *operands: Any,
        policy: ExecutionPolicy | None = None,
        **attributes: Any,
    ) -> DispatchExplanation:
        effective_policy = policy or ExecutionPolicy()
        definition, _, signature = self._prepare_call(op, tuple(operands), attributes)
        snapshot = self.snapshot()

        implementations = list(self.implementations.for_operator(definition.op_key))
        provider_rank = {
            provider: index for index, provider in enumerate(effective_policy.providers)
        }
        implementations.sort(
            key=lambda item: (
                provider_rank.get(item.manifest.provider, len(provider_rank)),
                -item.manifest.priority,
                item.manifest.implementation_id,
            )
        )

        candidates: list[CandidateEvaluation] = []
        selected: str | None = None
        for implementation in implementations:
            manifest = implementation.manifest
            reasons: list[str] = []
            if (
                effective_policy.implementation_id is not None
                and manifest.implementation_id != effective_policy.implementation_id
            ):
                reasons.append("not_requested_implementation")
            if (
                effective_policy.fixed_provider is not None
                and manifest.provider != effective_policy.fixed_provider
            ):
                reasons.append("not_requested_provider")
            if manifest.provider not in effective_policy.providers:
                reasons.append("provider_not_in_policy_order")

            declared = self._declared_support(manifest, signature, effective_policy)
            reasons.extend(declared.reasons)
            if not reasons and implementation.support is not None:
                custom = implementation.support(signature, effective_policy)
                reasons.extend(custom.reasons)
                if not custom.supported and not custom.reasons:
                    reasons.append("implementation_specific_rejection")

            capability = self._capabilities.get(manifest.implementation_id)
            evaluation = CandidateEvaluation(
                implementation_id=manifest.implementation_id,
                provider=manifest.provider,
                supported=not reasons,
                reasons=tuple(dict.fromkeys(reasons)),
                capability_status=capability.status if capability else "untested",
                release_tier=capability.release_tier if capability else "best_effort",
                build_fingerprint=manifest.build_fingerprint,
            )
            candidates.append(evaluation)
            if selected is None and evaluation.supported:
                selected = manifest.implementation_id

        return DispatchExplanation(
            op_key=str(definition.op_key),
            spec_revision=definition.spec_revision,
            signature_digest=signature.digest,
            snapshot_token=snapshot.token,
            selected_implementation_id=selected,
            candidates=tuple(candidates),
        )

    def supports(
        self,
        op: str,
        *operands: Any,
        policy: ExecutionPolicy | None = None,
        **attributes: Any,
    ) -> bool:
        return (
            self.explain_dispatch(op, *operands, policy=policy, **attributes)
            .selected_implementation_id
            is not None
        )

    def plan(
        self,
        op: str,
        *operands: Any,
        policy: ExecutionPolicy | None = None,
        **attributes: Any,
    ) -> ExecutionPlan:
        effective_policy = policy or ExecutionPolicy()
        definition, normalized, signature = self._prepare_call(
            op, tuple(operands), attributes
        )
        explanation = self.explain_dispatch(
            op, *operands, policy=effective_policy, **normalized
        )
        if explanation.selected_implementation_id is None:
            raise DispatchError(
                f"no eligible implementation for {definition.op_key}",
                details={"explanation": explanation.to_dict()},
            )
        implementation = self.implementations.get(
            explanation.selected_implementation_id
        )
        assert implementation is not None
        return ExecutionPlan(
            runtime=self,
            implementation=implementation,
            definition=definition,
            signature=signature,
            attributes=normalized,
            policy=effective_policy,
            snapshot_token=explanation.snapshot_token,
        )

    def execute(
        self,
        op: str,
        *operands: Any,
        policy: ExecutionPolicy | None = None,
        **attributes: Any,
    ) -> Any:
        plan = self.plan(op, *operands, policy=policy, **attributes)
        return plan.run(*operands, **attributes)
