"""Scope the permitted renderer's two inversions to immutable view caches.

SPDX-License-Identifier: Apache-2.0
No installed module or global torch function is modified. Cached values are
computed with the original inverse operators on the original inverse(c2w)
view matrix; the input camera_to_world is not used as an algebraic shortcut.
"""
from functools import update_wrapper
from types import FunctionType
import torch

CALLS = dict(renders=0, inverse=0, linalg_inv=0)


class _Proxy:
    def __init__(self, original, **overrides):
        self.original = original
        self.overrides = overrides

    def __getattr__(self, name):
        if name in self.overrides:
            return self.overrides[name]
        return getattr(self.original, name)


def reset_calls():
    CALLS.update(renders=0, inverse=0, linalg_inv=0)


def prepare_primary_inverse(world_to_camera):
    if world_to_camera.shape != (4, 4) or world_to_camera.requires_grad:
        raise ValueError('Fixed primary inverse needs one immutable view matrix')
    return torch.inverse(world_to_camera[None])


def rasterization_2dgs_cached_camera(original, *args, primary_inverse,
                                    linalg_inverse, **kwargs):
    if len(args) < 6:
        raise ValueError('Cached calibration needs the original positional camera ABI')
    view = args[5]
    if (view.ndim != 3 or view.shape != (1, 4, 4) or view.requires_grad
            or kwargs.get('packed') is not False or kwargs.get('render_mode') != 'RGB'
            or kwargs.get('sh_degree') is None):
        raise ValueError('Cached calibration requires one immutable unpacked RGB SH camera')
    for cached in (primary_inverse, linalg_inverse):
        if (cached is None or cached.shape != view.shape or cached.device != view.device
                or cached.dtype != view.dtype or cached.requires_grad):
            raise ValueError('Cached calibration matrices must match the immutable view')
    if (not isinstance(original, FunctionType) or original.__globals__.get('torch') is not torch
            or not {'inverse', 'inv'}.issubset(original.__code__.co_names)):
        raise RuntimeError('The permitted renderer fixed-inverse contract is unavailable')
    counts = dict(inverse=0, linalg_inv=0)

    def lookup(label, cached, matrix, *values, **keywords):
        # Producer dependency only: no numerical identity or byte comparison.
        if (values or keywords or matrix.shape != view.shape or matrix.dtype != view.dtype
                or matrix.device != view.device or matrix.data_ptr() != view.data_ptr()
                or matrix.stride() != view.stride()):
            raise RuntimeError('Cached inverse is not the current fixed view storage')
        counts[label] += 1
        if counts[label] != 1:
            raise RuntimeError('Unexpected duplicate fixed-camera inverse consumer')
        return cached

    def inverse(matrix, *values, **keywords):
        return lookup('inverse', primary_inverse, matrix, *values, **keywords)

    def linalg_inv(matrix, *values, **keywords):
        return lookup('linalg_inv', linalg_inverse, matrix, *values, **keywords)

    namespace = dict(original.__globals__)
    namespace['torch'] = _Proxy(torch, inverse=inverse,
        linalg=_Proxy(torch.linalg, inv=linalg_inv))
    cached_function = FunctionType(original.__code__, namespace, original.__name__,
        original.__defaults__, original.__closure__)
    update_wrapper(cached_function, original)
    cached_function.__kwdefaults__ = original.__kwdefaults__
    result = cached_function(*args, **kwargs)
    if counts != dict(inverse=1, linalg_inv=1):
        raise RuntimeError('Both original primary fixed-camera inverses must be consumed')
    CALLS['renders'] += 1
    for key, value in counts.items():
        CALLS[key] += value
    return result
