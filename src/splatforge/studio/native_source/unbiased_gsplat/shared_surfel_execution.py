"""One serial shared forward, with the original two autograd/backward paths.

SPDX-License-Identifier: Apache-2.0
The installed gsplat package is unchanged. Only the current process's native
forward is scoped; original RGB and auxiliary visibility remain independent.
"""
import importlib


class NativeSurfelSharedForward:
    def __init__(self, pgsr, surfel, *, native=None, last_backward=False, last_native=None):
        self._pgsr = pgsr
        self._surfel = surfel
        self._native = native
        if type(last_backward) is not bool:
            raise TypeError('Last-contributor backward must be boolean')
        self._last_backward = last_backward
        self._last_native = last_native
        self._entered = False
        self._active = False
        self._pending = None
        self._geometry = None
        self.calls = {}

    def _count(self, name):
        self.calls[name] = self.calls.get(name, 0) + 1

    def __enter__(self):
        from . import _cuda
        if (self._entered or _cuda.SHARED_SURFEL_FORWARD_EXECUTION is not None
                or _cuda.AUXILIARY_STREAM_EXECUTION is not None or _cuda.AUX_FRONT_PREFIX_FORWARD):
            raise RuntimeError('Shared Surfel forward requires its own original serial scope')
        native = self._native or importlib.import_module('shared_surfel_forward_cuda')
        if not getattr(native, 'independent_visibility_shared_forward', False):
            raise RuntimeError('The explicit shared Surfel forward backend is unavailable')
        self._native = native
        if self._last_backward:
            last = self._last_native or importlib.import_module('surfel_last_cuda')
            if not getattr(last, 'exact_last_contributor_batches', False):
                raise RuntimeError('The explicit last-contributor backward backend is unavailable')
            self._last_native = last
        self._owner = importlib.import_module('gsplat.csrc')
        self._original = self._owner.rasterize_to_pixels_2dgs_fwd
        if self._last_backward:
            self._original_backward = self._owner.rasterize_to_pixels_2dgs_bwd
            self._owner.rasterize_to_pixels_2dgs_bwd = self._backward
        self._owner.rasterize_to_pixels_2dgs_fwd = self._primary
        _cuda.SHARED_SURFEL_FORWARD_EXECUTION = self
        self._entered = True
        self._count('scope_enter')
        return self

    def pgsr(self, *args, **kwargs):
        if not self._entered or self._active:
            raise RuntimeError('PGSR requires an idle shared-forward scope')
        self._count('pgsr_render')
        return self._pgsr(*args, **kwargs)

    def surfel(self, *args, **kwargs):
        if not self._entered or self._active or self._pending is not None:
            raise RuntimeError('Shared Surfel forward is not ready for another render')
        if (not kwargs.get('surface_midpoint') or not kwargs.get('terminal_surface')
                or kwargs.get('ray_depth_order') or kwargs.get('auxiliary_refinement_gradient')):
            raise ValueError('Shared forward requires the current midpoint/terminal unmodified projection')
        auxiliary = kwargs.get('auxiliary', True)
        front = bool(auxiliary and kwargs.get('front_convergence', False))
        radius = float(kwargs.get('scene_radius', 1.))
        if front:
            radius = 4. / float(kwargs.get('convergence_physical_scale', 1.))
        self._geometry = dict(radius=radius, front=front)
        self._active = True
        self._count('surfel_render')
        try:
            value = self._surfel(*args, **kwargs)
            if self._pending is not None:
                raise RuntimeError('Shared primary forward had no matching auxiliary consumer')
            return value
        finally:
            self._pending = None
            self._geometry = None
            self._active = False

    def _primary(self, *args, **kwargs):
        if not self._active or self._pending is not None:
            raise RuntimeError('Shared primary forward requires its current Surfel consumer')
        if len(args) != 12 or kwargs:
            raise ValueError('Shared primary forward requires the qualified pinned native ABI')
        if args[5] is not None or args[6] is not None:
            raise ValueError('Shared primary forward requires no native background or tile mask')
        if args[0].ndim != 3 or args[0].shape[0] != 1 or args[2].shape[-1] != 3 or int(args[9]) != 16:
            raise ValueError('Shared primary forward requires one unpacked RGB3 tile16 image')
        geometry = self._geometry
        value = self._native.forward(*args, .1, .6, geometry['radius'], True, geometry['front'], True)
        if len(value) != 11:
            raise RuntimeError('Shared forward did not provide seven primary and four auxiliary fields')
        self._pending = dict(tensors=(args[0], args[1], args[3], args[10], args[11]),
            dimensions=tuple(int(x) for x in args[7:10]), auxiliary=tuple(value[7:]))
        self._count('shared_forward')
        return tuple(value[:7])

    def _backward(self, *args, **kwargs):
        if not self._entered or not self._last_backward:
            raise RuntimeError('Last-contributor backward requires its current render scope')
        self._count('last_contributor_backward')
        return self._last_native.rasterize_to_pixels_2dgs_bwd(*args, **kwargs)

    def auxiliary(self, arguments):
        pending = self._pending
        if not self._active or pending is None:
            raise RuntimeError('Shared auxiliary fields have no current primary producer')
        if tuple(arguments[5:8]) != pending['dimensions']:
            raise RuntimeError('Shared auxiliary image geometry changed')
        expected = (.1, .6, self._geometry['radius'], False, True, self._geometry['front'], True)
        if tuple(arguments[8:]) != expected:
            raise RuntimeError('Shared auxiliary estimator parameters changed')
        for actual, original in zip(arguments[:5], pending['tensors']):
            if (actual.device != original.device or actual.dtype != original.dtype
                    or actual.numel() != original.numel() or actual.data_ptr() != original.data_ptr()):
                raise RuntimeError('Shared auxiliary metadata is not the recorded projection storage')
        self._pending = None
        self._count('auxiliary_reuse')
        return pending['auxiliary']

    def __exit__(self, error_type, error, traceback):
        from . import _cuda
        pending = self._active or self._pending is not None
        try:
            self._owner.rasterize_to_pixels_2dgs_fwd = self._original
            if self._last_backward:
                self._owner.rasterize_to_pixels_2dgs_bwd = self._original_backward
        finally:
            _cuda.SHARED_SURFEL_FORWARD_EXECUTION = None
            self._entered = False
            self._active = False
            self._pending = None
            self._geometry = None
            self._count('scope_exit')
        if error_type is None and pending:
            raise RuntimeError('Shared-forward scope exited with an incomplete render')
        return False
