"""Explicitly registered, experimental tensor operators for surface methods.

This eager facade always names the Torch implementation. It never silently
selects an alternate renderer or claims trusted conformance evidence.
"""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files

import torch

from ..core.errors import SchemaValidationError
from ..core.models import CapabilityRecord, ImplementationManifest, OpDefinition, OpKey
from ..core.policy import ExecutionPolicy
from ..core.runtime import Runtime
from ..providers.torch.gaussian import reference
from ..providers.torch.gaussian.radegs import regularized_ray_geometry


_RUNNERS = {
    "gaussian.covariance@1": (reference.covariance, 2, {}),
    "gaussian.regularized_ray_geometry@1": (
        regularized_ray_geometry,
        5,
        {
            "width": None,
            "height": None,
            "relative_jitter": 1e-6,
            "absolute_jitter": 1e-8,
        },
    ),
    "gaussian.ray_peak@1": (reference.ray_peak, 4, {}),
    "camera.pixel_rays@1": (reference.pixel_rays, 1, {"width": None, "height": None}),
    "geometry.depth_normals@1": (reference.depth_normals, 2, {}),
    "gaussian.sh_color@1": (reference.sh_color, 2, {"degree": 3}),
    "gaussian.sh_ambiguity@1": (reference.sh_ambiguity, 1, {}),
    "gaussian.ray_color_variance@1": (reference.ray_color_variance, 3, {}),
    "splat.alpha_composite@1": (reference.alpha_composite, 4, {}),
    "splat.alpha_composite@2": (reference.alpha_composite_batch, 4, {}),
    "splat.alpha_composite@3": (
        reference.alpha_composite_batch_selected,
        4,
        {"distortions": None},
    ),
    "gaussian.ray_color_variance@2": (reference.ray_color_variance_batch, 3, {}),
    "gaussian.opacity_field@1": (reference.opacity_field, 5, {"gaussian_chunk": 1024}),
    "matching.patch_zncc@1": (reference.patch_zncc, 3, {"epsilon": 1e-8}),
}


def _validator(op, count, defaults):
    def validate(operands, attributes):
        if len(operands) != count or any(
            not isinstance(t, torch.Tensor) for t in operands
        ):
            raise SchemaValidationError(f"{op} expects {count} tensor operands")
        if any(t.dtype not in (torch.float32, torch.float64) for t in operands):
            raise SchemaValidationError("surface references support float32/float64")
        if any(
            (t.dtype, t.device) != (operands[0].dtype, operands[0].device)
            for t in operands
        ):
            raise SchemaValidationError("tensor dtype/device must match explicitly")
        if set(attributes) - set(defaults):
            raise SchemaValidationError("unknown operator attributes")
        attrs = {**defaults, **attributes}
        if "distortions" in attrs and attrs["distortions"] not in (
            "none",
            "l1",
            "l2",
            "both",
        ):
            raise SchemaValidationError(
                "distortions must explicitly be none, l1, l2 or both"
            )
        for name in ("width", "height", "gaussian_chunk"):
            if name in attrs and (type(attrs[name]) is not int or attrs[name] <= 0):
                raise SchemaValidationError(f"{name} must be a positive integer")
        if "degree" in attrs and (
            type(attrs["degree"]) is not int or not 0 <= attrs["degree"] <= 3
        ):
            raise SchemaValidationError("SH degree must be 0..3")
        if "epsilon" in attrs and not 0 < attrs["epsilon"] < 1:
            raise SchemaValidationError("epsilon must be in (0,1)")
        t = operands
        shapes_ok = True
        if op == "gaussian.regularized_ray_geometry@1":
            shapes_ok = (
                t[0].ndim == 2
                and t[0].shape[-1] == 3
                and t[1].shape == (t[0].shape[0], 4)
                and t[2].shape == t[0].shape
                and t[3].shape == (4, 4)
                and t[4].shape == (3, 3)
            )
            if attrs["relative_jitter"] <= 0 or attrs["absolute_jitter"] <= 0:
                raise SchemaValidationError("ray-geometry jitter must be positive")
        elif op == "gaussian.covariance@1":
            shapes_ok = (
                t[0].ndim == 2
                and t[0].shape[-1] == 4
                and t[1].shape == (t[0].shape[0], 3)
            )
        elif op == "gaussian.ray_peak@1":
            shapes_ok = (
                t[0].ndim == 2
                and t[0].shape[-1] == 3
                and t[1].shape == (t[0].shape[0], 3, 3)
                and t[2].ndim == 2
                and t[2].shape[-1] == 3
                and t[3].shape == t[2].shape
            )
        elif op == "camera.pixel_rays@1":
            shapes_ok = t[0].shape == (3, 3)
        elif op == "geometry.depth_normals@1":
            shapes_ok = t[0].ndim == 2 and t[1].shape == (3, 3)
        elif op.startswith("gaussian.sh_"):
            shapes_ok = t[0].ndim == 3 and t[0].shape[-1] == 3 and t[0].shape[1] >= 1
            if op == "gaussian.sh_color@1":
                shapes_ok = (
                    shapes_ok
                    and t[1].shape == (t[0].shape[0], 3)
                    and t[0].shape[1] >= (attrs["degree"] + 1) ** 2
                )
        elif op == "gaussian.ray_color_variance@1":
            shapes_ok = (
                t[0].ndim == 2
                and t[1].shape == (t[0].shape[0], 3)
                and t[2].shape == (t[0].shape[1], 3)
            )
        elif op == "splat.alpha_composite@1":
            shapes_ok = (
                t[0].ndim == 2
                and t[1].shape == (t[0].shape[0], 3)
                and t[2].shape == t[0].shape
                and t[3].shape == (*t[0].shape, 3)
            )
        elif op in ("splat.alpha_composite@2", "splat.alpha_composite@3"):
            shapes_ok = (
                t[0].ndim == 3
                and t[1].shape == (*t[0].shape[:2], 3)
                and t[2].shape == t[0].shape
                and t[3].shape == (*t[0].shape, 3)
            )
        elif op == "gaussian.ray_color_variance@2":
            shapes_ok = (
                t[0].ndim == 3
                and t[1].shape == (*t[0].shape[:2], 3)
                and t[2].shape == (t[0].shape[0], t[0].shape[2], 3)
            )
        elif op == "gaussian.opacity_field@1":
            shapes_ok = (
                t[0].ndim == 2
                and t[0].shape[-1] == 3
                and t[1].ndim == 2
                and t[1].shape[-1] == 3
                and t[2].shape == (t[1].shape[0], 3, 3)
                and t[3].shape == (t[1].shape[0],)
                and t[4].ndim == 2
                and t[4].shape[-1] == 3
                and t[4].shape[0] > 0
            )
        elif op == "matching.patch_zncc@1":
            shapes_ok = t[0].ndim >= 1 and t[0].shape == t[1].shape == t[2].shape
        if not shapes_ok:
            raise SchemaValidationError(f"invalid tensor shapes for {op}")
        return attrs

    return validate


def create_surface_runtime(
    *, implementations=(), capabilities=(), seal=True
) -> Runtime:
    contents = files(__package__).joinpath("semantics.json").read_bytes()
    revision = "sha256:" + hashlib.sha256(contents).hexdigest()
    spec = json.loads(contents)
    runtime = Runtime()
    for op, (runner, count, attrs) in _RUNNERS.items():
        if op not in spec["operators"]:
            raise RuntimeError(f"operator missing from semantic contract: {op}")
        identifier = "splatforge.torch." + op.replace("@", ".v")
        runtime.register_definition(
            OpDefinition(OpKey.parse(op), revision, _validator(op, count, attrs))
        )
        manifest = ImplementationManifest.from_dict(
            {
                "schema_version": 1,
                "implementation_id": identifier,
                "provider": "torch",
                "provider_api": 1,
                "op": op,
                "op_spec_revision": revision,
                "implementation_version": "0.1.0",
                "lowering_target": f"{runner.__module__}.{runner.__name__}",
                "static_variant_token": "reference",
                "priority": 0,
                "declared_eligibility": {
                    "devices": ["cpu", "cuda"],
                    "dtypes": ["float32", "float64"],
                    "layouts": ["contiguous", "strided_last_dim", "strided"],
                    "backward": True,
                    "jvp": False,
                    "determinism_guarantee": "none",
                    "host_synchronization": "none",
                    "cuda_graph": False,
                    "execution": {"eager": "supported"},
                },
                "build": {"fingerprint": revision},
            }
        )
        runtime.register_implementation(manifest, runner)
        runtime.record_capability(
            CapabilityRecord(
                identifier, "experimental", "surface-reference", "not-release-verified"
            )
        )
    for implementation in implementations:
        runtime.register_implementation(
            implementation.manifest,
            implementation.runner,
            support=implementation.support,
        )
    for capability in capabilities:
        runtime.record_capability(capability)
    if seal:
        runtime.seal()
    return runtime


class SurfaceOperators:
    """Explicit opt-in to experimental Torch semantics, suitable for compositions."""

    def __init__(self, runtime=None, *, implementations=None, allow_unverified=True):
        self.runtime = runtime or create_surface_runtime()
        self.implementations = dict(implementations or {})
        self.allow_unverified = allow_unverified

    def __call__(self, op, *tensors, **attributes):
        identifier = self.implementations.get(
            op, "splatforge.torch." + op.replace("@", ".v")
        )
        implementation = self.runtime.implementations.get(identifier)
        if implementation is None:
            raise ValueError(f"operator implementation is not registered: {identifier}")
        policy = ExecutionPolicy(
            implementation_id=identifier,
            providers=(implementation.manifest.provider,),
            allow_unverified=self.allow_unverified,
        )
        return self.runtime.execute(op, *tensors, policy=policy, **attributes)
