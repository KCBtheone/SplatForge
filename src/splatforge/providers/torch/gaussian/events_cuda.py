"""Fused GOF/AmbiSuR event equations + analytic first-order VJP.

Strided float32 CUDA operands supported; reductions use nondeterministic atomic
addition. Explicit provider, no fallback, no higher-order derivatives.
The support quadratic form retains the reference CUDA einsum contraction order;
its adjoint is returned through that node, preventing one-ULP boundary changes.
GOF also retains Qray/depth/exponent CUDA tensor nodes: cuBLAS contraction
order varies with batch shape, and tiny-alpha errors amplify normalized normals.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.autograd.function import once_differentiable

_extension = None
activation_counts={'prepared_forward':0,'prepared_backward':0}


def extension():
    global _extension
    if _extension is None:
        from torch.utils.cpp_extension import load

        root = os.environ.get("TORCH_EXTENSIONS_DIR")
        if not root:
            raise RuntimeError("Set TORCH_EXTENSIONS_DIR before loading the event CUDA extension")
        source = Path(__file__).with_suffix(".cu")
        # Explicit task-owned compiler cache; no source checksum/identity scan.
        from ...._paths import output_path

        build = output_path(Path(root).expanduser() / "surface_events_h016")
        build.mkdir(parents=True, exist_ok=True)
        _extension = load(
            name="sf_surface_events_h016",
            sources=[str(source)],
            build_directory=str(build),
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--fmad=false"],
            verbose=False,
        )
    return _extension


class _Events(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        delta,
        proxy,
        q2,
        q3,
        normals,
        distance,
        opacity,
        rays,
        squared,
        qray,
        ray_depth,
        exponent,
        valid,
        mode,
        near,
        far,
        extent,
    ):
        activation_counts['prepared_forward']+=1
        ctx.save_for_backward(
            delta,
            proxy,
            q2,
            q3,
            normals,
            distance,
            opacity,
            rays,
            squared,
            qray,
            ray_depth,
            exponent,
            valid,
        )
        ctx.settings = mode, near, far, extent
        return tuple(
            extension().forward(
                delta,
                proxy,
                q2,
                q3,
                normals,
                distance,
                opacity,
                rays,
                squared,
                qray,
                ray_depth,
                exponent,
                valid,
                *ctx.settings,
            )
        )

    @staticmethod
    @once_differentiable
    def backward(ctx, alpha, depth, normal):
        activation_counts['prepared_backward']+=1
        gradients = extension().backward(
            *ctx.saved_tensors, *ctx.settings, alpha, depth, normal
        )
        return (*gradients, None, None, None, None, None)


def surface_events_cuda(
    delta,
    proxy,
    q2,
    q3,
    normals,
    distance,
    opacity,
    rays,
    valid,
    *,
    mode,
    near,
    far,
    extent,
    q3_exponent=None,
):
    if mode not in ("gof", "ambisur"):
        raise ValueError("event CUDA only implements GOF/AmbiSuR")
    if any(
        x.device.type != "cuda" or x.dtype != torch.float32
        for x in (delta, proxy, q2, q3, normals, distance, opacity, rays)
    ):
        raise ValueError("explicit CUDA float32 event operands required")
    # This CUDA tensor contraction is intentional, not a fallback. Reassociating
    # it inside the event shader changes exact extent-boundary membership.
    squared = torch.einsum("tlpi,tlij,tlpj->tlp", delta, q2, delta)
    if mode == "gof":
        exponent_matrix = q3 if q3_exponent is None else q3_exponent
        if (
            exponent_matrix.device != q3.device
            or exponent_matrix.dtype != q3.dtype
            or exponent_matrix.shape != q3.shape
        ):
            raise ValueError("explicit matching GOF exponent-matrix node required")
        # Explicit arithmetic contract, not fallback or detached surrogate.
        # Preserve the actual CUDA einsum and pointwise reduction nodes, with
        # their original autograd VJPs. Fixed FMA order is not shape invariant.
        qray = torch.einsum("tlij,tpj->tlpi", q3, rays)
        denominator = (qray * rays[:, None]).sum(-1).clamp_min(1e-12)
        ray_depth = (proxy * qray).sum(-1) / denominator
        offset = ray_depth[..., None] * rays[:, None] - proxy
        exponent = torch.einsum("tlpi,tlij,tlpj->tlp", offset, exponent_matrix, offset)
    else:
        # Correct-rank borrowed operands, ignored by the Ambi shader branch;
        # its prepared-node adjoints are exact zeros, with no allocation/copy.
        qray, ray_depth, exponent = proxy, squared, squared
    return _Events.apply(
        delta,
        proxy,
        q2,
        q3,
        normals,
        distance,
        opacity,
        rays,
        squared,
        qray,
        ray_depth,
        exponent,
        valid,
        0 if mode == "gof" else 1,
        near,
        far,
        extent,
    )
