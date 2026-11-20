"""Skip only tiles with zero incoming VJP, using the original native kernels.

SPDX-License-Identifier: Apache-2.0
Forward visibility, all original backward arithmetic and screen stores remain.
"""
import importlib

_ACTIVE = None


class NativeAdjointTiles:
    def __init__(self, *, native=None):
        self._native = native
        self._entered = False
        self.calls = {}

    def __enter__(self):
        global _ACTIVE
        if self._entered or _ACTIVE is not None:
            raise RuntimeError('Adjoint tile scope is already active')
        native = self._native or importlib.import_module('adjoint_tiles_cuda')
        if not getattr(native, 'exact_zero_adjoint_tile_mask', False):
            raise RuntimeError('The explicit zero-adjoint tile backend is unavailable')
        owner = importlib.import_module('gsplat.csrc')
        self._native = native
        self._owner = owner
        self._pgsr = owner.rasterize_to_pixels_3dgs_bwd
        self._surfel = owner.rasterize_to_pixels_2dgs_bwd
        owner.rasterize_to_pixels_3dgs_bwd = self._pgsr_backward
        owner.rasterize_to_pixels_2dgs_bwd = self._surfel_backward
        self._entered = True
        _ACTIVE = self
        return self

    def _backward(self, label, original, args, kwargs):
        if not self._entered:
            raise RuntimeError('Adjoint tile backward is outside its render scope')
        surfel = label == 'surfel'
        count, mask_index, width_index, tile_index, gradient_start = (
            (23, 7, 8, 10, 17) if surfel else (16, 5, 6, 8, 13))
        if len(args) != count or kwargs:
            raise ValueError('Adjoint tiles require the qualified pinned native ABI')
        if args[mask_index] is not None or args[mask_index-1] is not None or args[-1]:
            raise ValueError('Adjoint tiles require the original no-mask/no-background/non-absgrad consumer')
        if (args[0].ndim != 3 or args[0].shape[0] != 1 or
                args[2].shape[-1] != (3 if surfel else 8) or int(args[tile_index]) != 16):
            raise ValueError('Adjoint tiles require original unpacked tile16 RGB3 or feature8 fields')
        gradients = args[gradient_start:-1]
        height, width = int(args[width_index+1]), int(args[width_index])
        if any(value.shape[:3] != (1, height, width) for value in gradients):
            raise ValueError('Adjoint image geometry changed')
        mask = self._native.mask(gradients)
        changed = list(args)
        changed[mask_index] = mask
        self.calls[label] = self.calls.get(label, 0) + 1
        return original(*changed)

    def _pgsr_backward(self, *args, **kwargs):
        return self._backward('pgsr', self._pgsr, args, kwargs)

    def _surfel_backward(self, *args, **kwargs):
        return self._backward('surfel', self._surfel, args, kwargs)

    def __exit__(self, error_type, error, traceback):
        global _ACTIVE
        try:
            self._owner.rasterize_to_pixels_3dgs_bwd = self._pgsr
            self._owner.rasterize_to_pixels_2dgs_bwd = self._surfel
        finally:
            self._entered = False
            _ACTIVE = None
        return False
