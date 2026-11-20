"""Optional physical-ray pixel-phase refinement demand; no parameter VJP change.

SPDX-License-Identifier: Apache-2.0
"""
from __future__ import annotations
import importlib
import os
from pathlib import Path
import torch


def phase4_magnitude(bins):
    """Sum the norms of sixteen SIGNED pixel-phase screen-gradient sums.

    Bins already include the original width/2 and height/2 screen scaling.
    This is not componentwise absgrad or an exact per-pixel norm sum.
    """
    if not isinstance(bins, torch.Tensor) or not bins.is_floating_point():
        raise TypeError('Phase4 bins must be a floating tensor')
    if bins.ndim != 4 or bins.shape[0] != 1 or bins.shape[-2:] != (16, 2):
        raise ValueError('Phase4 bins require unpacked [1,N,16,2]')
    return torch.linalg.vector_norm(bins.detach(), dim=-1).sum(dim=-1)


def phase4_view_statistic(screen, batch_size):
    if type(batch_size) is not int or batch_size != 1:
        raise ValueError('Phase4 currently requires the qualified single-view consumer')
    value = getattr(screen, 'phase4_gradient_magnitude', None)
    if (not isinstance(value, torch.Tensor) or value.shape != screen.shape[:-1]
            or value.device != screen.device or value.dtype != screen.dtype):
        raise RuntimeError('Missing or incompatible physical phase4 refinement statistic')
    return value[0].detach()


def validate_refinement_resume(previous_config, requested):
    if requested not in ('signed', 'phase4'):
        raise ValueError('Unknown physical refinement statistic')
    if previous_config.get('unbiased_refinement_statistic', 'signed') != requested:
        raise ValueError('Resume physical refinement statistic mismatch; start changed training from raw SfM')


def validate_phase4_consumer(args, kwargs):
    if len(args) != 23 or kwargs:
        raise ValueError('Phase4 requires the qualified native 2DGS backward ABI')
    means, rays, colors, opacity, normals, densify = args[:6]
    if (means.ndim != 3 or means.shape[0] != 1 or means.shape[-1] != 2
            or rays.shape != (*means.shape[:-1], 3, 3)
            or colors.shape != (*means.shape[:-1], 3)
            or normals.shape != colors.shape or opacity.shape != means.shape[:-1]
            or densify.shape != means.shape):
        raise ValueError('Phase4 requires unpacked single-image RGB3 fields')
    if args[6] is not None or args[7] is not None or args[-1] or args[10] != 16:
        raise ValueError('Phase4 requires tile16 without native background/mask/absgrad')
    tensors = [value for value in args if isinstance(value, torch.Tensor)]
    if (not means.is_cuda or means.dtype != torch.float32
            or any(value.device != means.device or not value.is_contiguous() for value in tensors)
            or any(value.is_floating_point() and value.dtype != torch.float32 for value in tensors)):
        raise ValueError('Phase4 requires contiguous CUDA float32 fields on one device')


def load_phase4_native():
    from torch.utils.cpp_extension import load
    import gsplat
    source = Path(__file__).resolve().parent
    package = Path(gsplat.__file__).resolve().parent
    gsplat_source = os.environ.get('GSPLAT_SOURCE_ROOT')
    if not gsplat_source:
        raise RuntimeError('Set GSPLAT_SOURCE_ROOT to the gsplat checkout for the phase4 extension')
    pinned = Path(gsplat_source).expanduser().resolve()
    cache = os.environ.get('TORCH_EXTENSIONS_DIR')
    if not cache:
        raise RuntimeError('Phase4 native build requires an explicit data-disk TORCH_EXTENSIONS_DIR')
    build = Path(cache) / 'physical_refinement_native'
    build.mkdir(parents=True, exist_ok=True)
    return load(name='forge_physical_phase4',
                sources=[str(source/'physical_refinement.cpp'), str(source/'physical_refinement.cu')],
                extra_include_paths=[str(package/'cuda/csrc'), str(pinned/'gsplat/cuda/include'),
                                     str(pinned/'gsplat/cuda/csrc'), str(pinned/'gsplat/cuda/csrc/third_party/glm')],
                extra_cflags=['-O3'], extra_cuda_cflags=['-O3', '-lineinfo', '-use_fast_math'],
                build_directory=str(build), verbose=False)


class NativePhysicalRefinement:
    """Process-local native hook; default host recipe never enters this scope."""
    def __init__(self, *, native=None, owner=None):
        self.native = native
        self.owner = owner
        self.enabled = True
        self.calls = 0
        self._entered = False

    def __enter__(self):
        if self._entered:
            raise RuntimeError('Physical refinement scope already entered')
        if self.owner is None:
            self.owner = importlib.import_module('gsplat.cuda._backend')._C
        if self.owner is None:
            raise RuntimeError('Qualified gsplat native backend is unavailable')
        self.original = self.owner.rasterize_to_pixels_2dgs_bwd
        if getattr(self.original, '_forge_phase4', False):
            raise RuntimeError('Nested physical refinement scopes are unsupported')
        if self.native is None:
            self.native = load_phase4_native()
        def backward(*args, **kwargs):
            if not self.enabled:
                return self.original(*args, **kwargs)
            validate_phase4_consumer(args, kwargs)
            result = self.native.backward(*args)
            if len(result) != 8:
                raise RuntimeError('Physical phase4 native output ABI mismatch')
            args[5].phase4_gradient_magnitude = phase4_magnitude(result[7])
            self.calls += 1
            return tuple(result[:7])
        backward._forge_phase4 = True
        self.owner.rasterize_to_pixels_2dgs_bwd = backward
        self._entered = True
        return self

    def __exit__(self, error_type, error, traceback):
        if self._entered:
            self.owner.rasterize_to_pixels_2dgs_bwd = self.original
            self._entered = False
        return False
