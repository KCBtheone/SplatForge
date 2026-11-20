"""Explicit fused stable RaDe-GS geometry CUDA backend; no implicit fallback.

Float32 contiguous API, double internal Cholesky, analytic first derivatives.
Fixed-camera VJP is one generated reverse sweep, not ten directional replays.
Camera gradients use nondeterministic atomic summation; second derivatives are
not supported. Visibility is discrete and caller supplied. Invisible rows are
zero and are never evaluated (including invalid camera-z).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch
from torch.autograd.function import once_differentiable

_extension = None


def extension():
    global _extension
    if _extension is None:
        from torch.utils.cpp_extension import load

        root = os.environ.get("TORCH_EXTENSIONS_DIR")
        if not root:
            raise RuntimeError("Set TORCH_EXTENSIONS_DIR before loading the ray geometry CUDA extension")
        source = Path(__file__).with_suffix(".cu")
        # Generated fixed-camera reverse equations are part of the binary ID.
        token = hashlib.sha256(
            source.read_bytes()
            + source.with_name("ray_geometry_reverse.cuh").read_bytes()
        ).hexdigest()[:12]
        from ...._paths import output_path

        build = output_path(Path(root).expanduser() / ("ray_geometry_" + token))
        build.mkdir(parents=True, exist_ok=True)
        _extension = load(
            name="sf_ray_geometry_" + token,
            sources=[str(source)],
            build_directory=str(build),
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--fmad=false"],
            verbose=False,
        )
    return _extension


class _Geometry(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means,
        quats,
        scales,
        transform,
        K,
        visible,
        width,
        height,
        relative,
        absolute,
    ):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(means, quats, scales, transform, K, visible)
        ctx.settings = width, height, relative, absolute
        ctx.cameras = transform.requires_grad or K.requires_grad
        return tuple(
            extension().forward(
                means, quats, scales, transform, K, visible, *ctx.settings
            )
        )

    @staticmethod
    @once_differentiable
    def backward(ctx, distance, planes, normals):
        # Autograd may supply expanded/strided adjoints. Materialize explicitly
        # at this VJP boundary; forward operands are never silently copied.
        gradients = tuple(
            x.contiguous() if x is not None else None
            for x in (distance, planes, normals)
        )
        values = extension().backward(
            *ctx.saved_tensors, *ctx.settings, *gradients, ctx.cameras
        )
        return (*values, None, None, None, None, None)


def regularized_ray_geometry_cuda(
    means,
    quats,
    scales,
    world_to_camera,
    K,
    *,
    width,
    height,
    visible=None,
    relative_jitter=1e-6,
    absolute_jitter=1e-8,
):
    operands = means, quats, scales, world_to_camera, K
    if any(
        x.device.type != "cuda" or x.dtype != torch.float32 or not x.is_contiguous()
        for x in operands
    ):
        raise ValueError("explicit contiguous CUDA float32 operands required")
    if any(x.device != means.device for x in operands):
        raise ValueError("operands must use one CUDA device")
    if width <= 0 or height <= 0 or relative_jitter < 0 or absolute_jitter <= 0:
        raise ValueError("invalid geometry settings")
    if visible is None:
        visible = torch.ones(len(means), device=means.device, dtype=torch.bool)
    return _Geometry.apply(
        *operands, visible, width, height, relative_jitter, absolute_jitter
    )
