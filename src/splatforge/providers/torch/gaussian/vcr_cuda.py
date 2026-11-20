"""CUDA rasterization of VCR-GauS tangent-plane depth, normals and distortion."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from ...._paths import data_root, output_path

_SOURCE = Path(__file__).resolve().parent
_EXTENSION = None


def extension():
    """Load (and compile on first use) the native rasterization extension."""
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    from torch.utils.cpp_extension import load

    root = os.environ.get("TORCH_EXTENSIONS_DIR") or os.environ.get("SPLATFORGE_CUDA_BUILD_ROOT")
    build = None
    if root:
        build = output_path(Path(root) / "splatforge_vcr_cuda")
        build.mkdir(parents=True, exist_ok=True)
    elif data_root() is not None:
        raise RuntimeError("Set TORCH_EXTENSIONS_DIR to a directory inside SPLATFORGE_DATA_ROOT")
    _EXTENSION = load(
        name="splatforge_vcr_cuda",
        sources=[
            str(_SOURCE / "csrc" / "vcr_rasterizer.cpp"),
            str(_SOURCE / "csrc" / "vcr_launch.cu"),
            str(_SOURCE / "csrc" / "vcr_forward.cu"),
            str(_SOURCE / "csrc" / "vcr_backward.cu"),
        ],
        extra_include_paths=[str(_SOURCE / "csrc")],
        extra_cflags=["/O2"] if os.name == "nt" else ["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        build_directory=str(build) if build is not None else None,
        verbose=False,
    )
    return _EXTENSION


def _check(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if not tensor.is_cuda:
        raise ValueError(f"{name} must live on the CUDA device")
    if tensor.dtype != torch.float32:
        raise ValueError(f"{name} must be float32")
    return tensor.contiguous()


class _Rasterize(torch.autograd.Function):
    """Autograd bridge around the native rasterization kernels."""

    @staticmethod
    def forward(
        ctx,
        means2d,
        means,
        scales,
        rotations,
        opacities,
        colours,
        normals,
        dirs,
        view,
        proj,
        background,
        focal_x,
        focal_y,
        tan_fov_x,
        tan_fov_y,
        width,
        height,
    ):
        outputs = extension().rasterize_forward(
            means.contiguous(),
            scales.contiguous(),
            rotations.contiguous(),
            opacities.contiguous(),
            colours.contiguous(),
            normals.contiguous(),
            dirs.contiguous(),
            view.contiguous(),
            proj.contiguous(),
            background.contiguous(),
            float(focal_x),
            float(focal_y),
            float(tan_fov_x),
            float(tan_fov_y),
            int(width),
            int(height),
        )
        view = view.contiguous()
        proj = proj.contiguous()
        ctx.save_for_backward(
            means,
            scales,
            rotations,
            opacities,
            colours,
            normals,
            dirs,
            view,
            proj,
            background,
            outputs[18],  # ranges
            outputs[19],  # point list
            outputs[9],   # radii
            outputs[10],  # screen x
            outputs[11],  # screen y
            outputs[12],  # view x
            outputs[13],  # view y
            outputs[14],  # view z
            outputs[15],  # conic a
            outputs[16],  # conic b
            outputs[17],  # conic c
            outputs[5],   # final transmittance
            outputs[6],   # moment one
            outputs[7],   # moment two
            outputs[8],   # contributors
            outputs[3].detach(),  # rendered alpha
            outputs[1].detach(),  # rendered depth
        )
        ctx.settings = (focal_x, focal_y, tan_fov_x, tan_fov_y, width, height)
        return (
            outputs[0],
            outputs[1],
            outputs[2],
            outputs[3],
            outputs[4],
            outputs[9],
        )

    @staticmethod
    def backward(ctx, grad_colour, grad_depth, grad_normal, grad_alpha,
                 grad_distortion, grad_radii):
        (
            means,
            scales,
            rotations,
            opacities,
            colours,
            normals,
            dirs,
            view,
            proj,
            background,
            ranges,
            point_list,
            radii,
            screen_x,
            screen_y,
            view_x,
            view_y,
            view_z,
            conic_a,
            conic_b,
            conic_c,
            final_transmittance,
            moment_one,
            moment_two,
            contributors,
            rendered_alpha,
            rendered_depth,
        ) = ctx.saved_tensors
        focal_x, focal_y, tan_fov_x, tan_fov_y, width, height = ctx.settings
        grads = extension().rasterize_backward(
            ranges,
            point_list,
            radii,
            screen_x,
            screen_y,
            view_x,
            view_y,
            view_z,
            conic_a,
            conic_b,
            conic_c,
            means,
            scales,
            rotations,
            opacities,
            colours,
            normals,
            dirs,
            background,
            final_transmittance,
            moment_one,
            moment_two,
            contributors,
            rendered_alpha,
            rendered_depth,
            _check(grad_colour, "grad_colour"),
            _check(grad_depth, "grad_depth"),
            _check(grad_normal, "grad_normal"),
            _check(grad_alpha, "grad_alpha"),
            _check(grad_distortion, "grad_distortion"),
            view,
            proj,
            focal_x,
            focal_y,
            tan_fov_x,
            tan_fov_y,
            int(width),
            int(height),
        )
        grad_means, grad_scales, grad_rotations, grad_opacity, grad_colours, \
            grad_normals, grad_mean2d, grad_mean2d_densify = grads
        return (
            # The reference renderer passes the same tensor as means2D and
            # means2D_densify, so autograd accumulates both the signed and the
            # absolute screen-space gradient into ``projected_means.grad``.
            # The studio densifier reads exactly that tensor.
            grad_mean2d[:, :2] + grad_mean2d_densify[:, :2],
            grad_means,
            grad_scales,
            grad_rotations,
            grad_opacity,
            grad_colours,
            grad_normals,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def rasterize(
    means2d,
    means,
    scales,
    rotations,
    opacities,
    colours,
    normals,
    dirs,
    view,
    proj,
    background,
    focal_x,
    focal_y,
    tan_fov_x,
    tan_fov_y,
    width,
    height,
):
    """Rasterize a scene into colour, depth, normal, alpha, distortion, radii."""
    return _Rasterize.apply(
        means2d,
        means,
        scales,
        rotations,
        opacities,
        colours,
        normals,
        dirs,
        view,
        proj,
        background,
        focal_x,
        focal_y,
        tan_fov_x,
        tan_fov_y,
        width,
        height,
    )


def count(
    means,
    scales,
    rotations,
    opacities,
    view,
    proj,
    focal_x,
    focal_y,
    tan_fov_x,
    tan_fov_y,
    width,
    height,
):
    """Per Gaussian pixel activation counts (visibility / importance)."""
    with torch.no_grad():
        return extension().rasterize_counts(
            means.contiguous(),
            scales.contiguous(),
            rotations.contiguous(),
            opacities.contiguous(),
            view.contiguous(),
            proj.contiguous(),
            float(focal_x),
            float(focal_y),
            float(tan_fov_x),
            float(tan_fov_y),
            int(width),
            int(height),
        )
