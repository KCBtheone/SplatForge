"""Opt-in CUDA float32 implementation of selected compositing; never falls back."""

from __future__ import annotations

import hashlib
import os
from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace

import torch

from ....core.models import CapabilityRecord, ImplementationManifest

_extension = None


def extension():
    global _extension
    if _extension is None:
        from torch.utils.cpp_extension import load

        root = os.environ.get("TORCH_EXTENSIONS_DIR")
        if not root:
            raise RuntimeError(
                "set explicit task-owned TORCH_EXTENSIONS_DIR before CUDA build"
            )
        build = Path(root).resolve() / "selected_composite_v1"
        if os.path.isdir("/root/autodl-tmp") and not build.is_relative_to(
            Path("/root/autodl-tmp")
        ):
            raise ValueError("CUDA build must remain on the AutoDL data disk")
        build.mkdir(parents=True, exist_ok=True)
        _extension = load(
            name="sf_selected_composite_v1",
            sources=[str(Path(__file__).with_name("composite_cuda.cu"))],
            build_directory=str(build),
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    return _extension


class _SelectedComposite(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, color, depth, normal, mode):
        ctx.set_materialize_grads(False)
        order = (
            depth.sort(1).indices
            if mode & 1
            else torch.empty(
                (a.shape[0], 0, a.shape[2]), device=a.device, dtype=torch.int64
            )
        )
        outputs = extension().forward(a, color, depth, normal, order, mode)
        ctx.mode = mode
        ctx.save_for_backward(
            a,
            color,
            depth,
            normal,
            outputs[7],
            outputs[8],
            outputs[9],
            outputs[10],
            outputs[11],
            outputs[12],
            outputs[13],
        )
        return tuple(outputs[:8])

    @staticmethod
    def backward(ctx, *gradients):
        values = extension().backward(*ctx.saved_tensors, *gradients, ctx.mode)
        return *values, None


def composite(a, color, depth, normal, *, distortions):
    if a.device.type != "cuda" or a.dtype != torch.float32:
        raise ValueError("selected CUDA composite requires explicit CUDA float32")
    if a.numel() >= 2**31:
        raise ValueError(
            "selected CUDA composite uses 32-bit accessors; split the execution batch"
        )
    mode = {"none": 0, "l1": 1, "l2": 2, "both": 3}[distortions]
    return _SelectedComposite.apply(a, color, depth, normal, mode)


def implementation():
    revision = (
        "sha256:"
        + hashlib.sha256(
            files("splatforge.gaussian").joinpath("semantics.json").read_bytes()
        ).hexdigest()
    )
    identifier = "splatforge.cuda.selected_composite.v1"
    manifest = ImplementationManifest.from_dict(
        {
            "schema_version": 1,
            "implementation_id": identifier,
            "provider": "cuda",
            "provider_api": 1,
            "op": "splat.alpha_composite@3",
            "op_spec_revision": revision,
            "implementation_version": "0.1.0",
            "lowering_target": __name__ + ".composite",
            "static_variant_token": "cuda-float32-selected",
            "priority": 0,
            "declared_eligibility": {
                "devices": ["cuda"],
                "dtypes": ["float32"],
                "layouts": ["contiguous", "strided_last_dim", "strided"],
                "backward": True,
                "jvp": False,
                "determinism_guarantee": "none",
                "host_synchronization": "none",
                "cuda_graph": False,
                "execution": {"eager": "supported"},
            },
            "build": {
                "fingerprint": "sha256:"
                + hashlib.sha256(
                    Path(__file__).with_name("composite_cuda.cu").read_bytes()
                ).hexdigest()
            },
        }
    )
    return SimpleNamespace(
        manifest=manifest, runner=composite, support=None
    ), CapabilityRecord(
        identifier,
        "experimental",
        "cuda-composite",
        "requires-independent-value-gradient-gates",
    )
