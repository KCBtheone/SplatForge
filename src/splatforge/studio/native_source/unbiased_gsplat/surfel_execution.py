"""Opt-in host execution of permitted gsplat; tensor operators are unchanged.

SPDX-License-Identifier: Apache-2.0
The original backward remains owned by the installed Apache-2.0 gsplat module.
Only its function-local explicit host synchronization is suppressed. The real
torch module and all synchronization outside that function are untouched.
"""
from __future__ import annotations

import importlib
from functools import update_wrapper
from types import FunctionType
import torch

SUPPRESSED_SURFEL_SYNCHRONIZATIONS = 0


class _AttributeProxy:
    def __init__(self, target, **overrides):
        self._target = target
        self._overrides = overrides

    def __getattr__(self, name):
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._target, name)


def _suppress_local_sync(*args, **kwargs):
    global SUPPRESSED_SURFEL_SYNCHRONIZATIONS
    if args or kwargs:
        raise RuntimeError('Unsupported gsplat backward synchronization signature')
    SUPPRESSED_SURFEL_SYNCHRONIZATIONS += 1


def configure_async_surfel_backward(enabled):
    """Enable or undo only this process-local gsplat backward execution option."""
    if type(enabled) is not bool:
        raise TypeError('Async Surfel backward must be boolean')
    wrapper = importlib.import_module('gsplat.cuda._wrapper')
    owner = wrapper._RasterizeToPixels2DGS
    current = owner.backward
    original = getattr(current, '_forge_original_backward', None)
    if not enabled:
        if original is not None:
            owner.backward = staticmethod(original)
        return
    if original is not None:
        return
    if (current.__module__ != 'gsplat.cuda._wrapper'
            or current.__globals__.get('torch') is not torch
            or 'synchronize' not in current.__code__.co_names
            or 'cuda' not in current.__code__.co_names):
        raise RuntimeError('Unsupported gsplat 2DGS backward execution contract')
    namespace = dict(current.__globals__)
    namespace['torch'] = _AttributeProxy(torch,
        cuda=_AttributeProxy(torch.cuda, synchronize=_suppress_local_sync))
    asynchronous = FunctionType(current.__code__, namespace, current.__name__,
        current.__defaults__, current.__closure__)
    update_wrapper(asynchronous, current)
    asynchronous.__kwdefaults__ = current.__kwdefaults__
    asynchronous._forge_original_backward = current
    owner.backward = staticmethod(asynchronous)


class NativeFamilyStreams:
    """The qualified PGSR/Surfel fork-join schedule within one training scope."""

    def __init__(self, pgsr, surfel):
        self._pgsr = pgsr
        self._surfel = surfel
        self._streams = None
        self._pending = None
        self.calls = {}

    def _count(self, label):
        self.calls[label] = self.calls.get(label, 0) + 1

    @staticmethod
    def _record(value, stream):
        if isinstance(value, torch.Tensor):
            if value.is_cuda:
                value.record_stream(stream)
        elif isinstance(value, dict):
            for part in value.values():
                NativeFamilyStreams._record(part, stream)
        elif isinstance(value, (list, tuple)):
            for part in value:
                NativeFamilyStreams._record(part, stream)

    def __enter__(self):
        if self._streams is not None:
            raise RuntimeError('Family stream scope is already active')
        wrapper = importlib.import_module('gsplat.cuda._wrapper')
        if getattr(wrapper._RasterizeToPixels2DGS.backward, '_forge_original_backward', None) is not None:
            raise RuntimeError('Family stream scope requires the original backward at entry')
        streams = {}
        for family in ('pgsr', 'surfel'):
            streams[family] = torch.cuda.Stream()
            self._count('stream_create')
        configure_async_surfel_backward(True)
        self._streams = streams
        return self

    def pgsr(self, *args, **kwargs):
        if self._streams is None or self._pending is not None:
            raise RuntimeError('PGSR requires an active unpaired family stream scope')
        origin = torch.cuda.current_stream()
        stream = self._streams['pgsr']
        stream.wait_stream(origin)
        self._record((args, kwargs), stream)
        # Register the producer before calling it so error exit also joins it.
        self._pending = dict(origin=origin, value=None)
        with torch.cuda.stream(stream):
            self._count('stream_pgsr_render')
            value = self._pgsr(*args, **kwargs)
        self._pending['value'] = value
        return value

    def surfel(self, *args, **kwargs):
        if self._streams is None or self._pending is None:
            raise RuntimeError('Surfel render requires the paired PGSR producer')
        origin = torch.cuda.current_stream()
        if origin != self._pending['origin']:
            raise RuntimeError('Family render pair changed its producer stream')
        stream = self._streams['surfel']
        stream.wait_stream(origin)
        self._record((args, kwargs), stream)
        with torch.cuda.stream(stream):
            self._count('stream_surfel_render')
            value = self._surfel(*args, **kwargs)
        for stream in self._streams.values():
            origin.wait_stream(stream)
        self._record(self._pending['value'], origin)
        self._record(value, origin)
        self._pending = None
        self._count('pair_join')
        return value

    def __exit__(self, error_type, error, traceback):
        pending = self._pending
        try:
            if pending is not None:
                for stream in self._streams.values():
                    pending['origin'].wait_stream(stream)
                self._record(pending['value'], pending['origin'])
        finally:
            configure_async_surfel_backward(False)
            self._pending = None
            self._streams = None
        if pending is not None and error_type is None:
            raise RuntimeError('Unpaired PGSR render at family stream scope exit')
        return False


class NativeSurfelAuxiliaryStreams:
    """Fork primary and owned auxiliary rasterization after shared projection."""

    def __init__(self, pgsr, surfel):
        self._pgsr = pgsr
        self._surfel = surfel
        self._stream = None
        self._origin = None
        self._pending = None
        self._active = False
        self._auxiliary_started = False
        self.calls = {}

    def _count(self, label):
        self.calls[label] = self.calls.get(label, 0) + 1

    def __enter__(self):
        from . import _cuda
        if self._stream is not None or _cuda.AUXILIARY_STREAM_EXECUTION is not None:
            raise RuntimeError('Surfel auxiliary stream scope is already active')
        wrapper = importlib.import_module('gsplat.cuda._wrapper')
        owner = wrapper._RasterizeToPixels2DGS
        if getattr(owner.backward, '_forge_original_backward', None) is not None:
            raise RuntimeError('Surfel auxiliary scope requires the original backward at entry')
        original = owner.forward
        if getattr(original, '_forge_projection_auxiliary_original', None) is not None:
            raise RuntimeError('Surfel projection scope is already installed')
        self._owner = owner
        self._forward = original
        stream = torch.cuda.Stream()
        self._count('stream_create')

        def primary(ctx, means2d, ray_transforms, colors, opacities, normals,
                    densify, backgrounds, masks, width, height, tile_size,
                    isect_offsets, flatten_ids, absgrad, distloss):
            if not self._active or self._pending is not None:
                raise RuntimeError('Unexpected or unconsumed primary Surfel projection')
            origin = torch.cuda.current_stream()
            if origin != self._origin:
                raise RuntimeError('Primary Surfel projection changed its caller stream')
            ready = torch.cuda.Event()
            ready.record(origin)
            self._count('projection_ready')
            self._pending = dict(event=ready, origin=origin,
                tensors=(means2d, ray_transforms, opacities, isect_offsets, flatten_ids),
                geometry=(int(width), int(height), int(tile_size)))
            return original(ctx, means2d, ray_transforms, colors, opacities, normals,
                densify, backgrounds, masks, width, height, tile_size,
                isect_offsets, flatten_ids, absgrad, distloss)

        update_wrapper(primary, original)
        primary._forge_projection_auxiliary_original = original
        configure_async_surfel_backward(True)
        owner.forward = staticmethod(primary)
        self._stream = stream
        _cuda.AUXILIARY_STREAM_EXECUTION = self
        return self

    def pgsr(self, *args, **kwargs):
        if self._stream is None or self._active:
            raise RuntimeError('PGSR requires an active idle auxiliary stream scope')
        self._count('pgsr_render')
        return self._pgsr(*args, **kwargs)

    def surfel(self, *args, **kwargs):
        if self._stream is None or self._active or self._pending is not None:
            raise RuntimeError('Surfel auxiliary scope is not ready for another render')
        self._origin = torch.cuda.current_stream()
        self._active = True
        try:
            self._count('surfel_render')
            value = self._surfel(*args, **kwargs)
            if self._pending is not None:
                raise RuntimeError('Primary Surfel projection had no auxiliary consumer')
            return value
        finally:
            if self._auxiliary_started:
                self._origin.wait_stream(self._stream)
            self._auxiliary_started = False
            self._pending = None
            self._active = False
            self._origin = None

    def auxiliary(self, function, arguments):
        pending = self._pending
        if self._stream is None or not self._active or pending is None:
            raise RuntimeError('Auxiliary raster requires its current primary projection')
        origin = torch.cuda.current_stream()
        if origin != pending['origin'] or tuple(arguments[5:8]) != pending['geometry']:
            raise RuntimeError('Auxiliary raster changed projection stream or image geometry')
        # This checks the producer dependency, not numeric array identity. A
        # post-primary copy would need a later event and cannot use this fork.
        for actual, expected in zip(arguments[:5], pending['tensors']):
            if (not actual.is_cuda or actual.device != expected.device or
                    actual.dtype != expected.dtype or actual.numel() != expected.numel() or
                    actual.data_ptr() != expected.data_ptr()):
                raise RuntimeError('Auxiliary metadata is not the recorded projection storage')
        self._pending = None
        self._stream.wait_event(pending['event'])
        self._count('auxiliary_wait')
        NativeFamilyStreams._record(arguments, self._stream)
        self._auxiliary_started = True
        value = None
        try:
            with torch.cuda.stream(self._stream):
                self._count('auxiliary_dispatch')
                value = function(*arguments)
            return value
        finally:
            origin.wait_stream(self._stream)
            self._count('auxiliary_join')
            NativeFamilyStreams._record(value, origin)
            self._auxiliary_started = False

    def __exit__(self, error_type, error, traceback):
        from . import _cuda
        try:
            if self._auxiliary_started and self._origin is not None:
                self._origin.wait_stream(self._stream)
        finally:
            self._owner.forward = staticmethod(self._forward)
            configure_async_surfel_backward(False)
            _cuda.AUXILIARY_STREAM_EXECUTION = None
            self._stream = None
            self._origin = None
            self._pending = None
            self._active = False
            self._auxiliary_started = False
        return False
