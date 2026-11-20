"""Explicit C++ host execution of the existing AmbiSuR ATen graph."""

from collections import defaultdict
import importlib.util
import os
from pathlib import Path

import torch

_extension = None


def extension(*, compile=False):
    global _extension
    if _extension is None:
        root = os.environ.get("SF_AMBISUR_ATEN_BUILD_DIR")
        if not root:
            raise RuntimeError("explicit task-owned SF_AMBISUR_ATEN_BUILD_DIR required")
        from .._paths import output_path

        build = output_path(root)
        if compile:
            from torch.utils.cpp_extension import load

            build.mkdir(parents=True, exist_ok=True)
            scratch = build / "tmp"
            scratch.mkdir(exist_ok=True)
            os.environ["TMPDIR"] = str(scratch)
            _extension = load(name="sf_ambisur_aten_host", sources=[str(Path(__file__).with_suffix(".cpp"))],
                build_directory=str(build), extra_cflags=["-O3"], with_cuda=False, verbose=False)
        else:
            binary = build / "sf_ambisur_aten_host.so"
            if not binary.is_file():
                raise RuntimeError("precompile and qualify the task-owned ATen module before execution")
            spec = importlib.util.spec_from_file_location("sf_ambisur_aten_host", binary)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _extension = module
    return _extension


class AtenCenterLayoutEvents:
    def __init__(self):
        self.counts = defaultdict(int)

    def __call__(self, *args):
        self.counts["producer_calls"] += 1
        return tuple(extension().events(*args))


class AtenCompositeOperators:
    def __init__(self, base):
        self.base = base
        self.counts = defaultdict(int)

    def __call__(self, name, *args, **kwargs):
        if name != "splat.alpha_composite@3":
            return self.base(name, *args, **kwargs)
        self.counts["composite_calls"] += 1
        return tuple(extension().composite(*args, kwargs["distortions"],
            kwargs.get("detach_l1_weights", False)))


def check_extension(device="cpu"):
    """Real compiled operator/adjoint fixtures, not a scene or speed result."""
    from .ambisur_events import center_layout_events
    from ..providers.torch.gaussian import reference
    from torch.utils.checkpoint import checkpoint

    module = extension()
    ids = torch.tensor([[0, 1, 2], [2, 0, 1]], device=device)
    valid = torch.tensor([[True, True, False], [True, False, False]], device=device)
    pixels = torch.tensor([[[.5, .5], [1.5, .5], [.5, 1.5]],
                           [[20., 20.], [21., 20.], [20., 21.]]], device=device)
    rays = torch.tensor([.1, .2, 1.], device=device).expand(2, 3, 3)
    def inputs():
        return [v.clone().detach().requires_grad_() for v in (
            torch.tensor([[.5, .5], [1.2, .7], [40., 40.]], device=device),
            torch.tensor([[[1., .2], [.2, 2.]], [[2., -.1], [-.1, 1.]],
                          [[1., 0.], [0., 1.]]], device=device),
            torch.tensor([.2, .6, .4], device=device),
            torch.tensor([[0., 0., 1.], [.2, .1, 1.], [0., 0., 1.]], device=device),
            torch.tensor([2., 3., 4.], device=device))]
    maximum = 0.
    for empty in (False, True):
        a, b = inputs(), inputs()
        gate = torch.zeros_like(valid) if empty else valid
        expected = center_layout_events(ids, gate, rays, pixels, *a, 2., .01, 100., counts=defaultdict(int))
        actual = checkpoint(lambda *values:tuple(module.events(ids, gate, rays, pixels,
            *values, 2., .01, 100.)), *b, use_reentrant=False)
        for x, y in zip(expected, actual):
            torch.testing.assert_close(x, y, atol=3e-5, rtol=3e-4)
            maximum = max(maximum, float((x-y).detach().abs().max()))
        for number in range(3):
            ga = torch.autograd.grad(expected[number].sum(), a, retain_graph=True, allow_unused=True)
            gb = torch.autograd.grad(actual[number].sum(), b, retain_graph=True, allow_unused=True)
            for x, y, p, q in zip(ga, gb, a, b):
                x = torch.zeros_like(p) if x is None else x
                y = torch.zeros_like(q) if y is None else y
                torch.testing.assert_close(x, y, atol=3e-4, rtol=3e-3)
        for mode in ("none", "l1", "l2", "both"):
            for detached in (False, True):
                aa = [v.detach().clone().requires_grad_() for v in (
                    expected[0], torch.tensor([[[.2, .3, .4], [.8, .5, .7], [.1, .2, .3]]],
                                               device=device).expand(2, -1, -1), expected[1], expected[2])]
                bb = [v.detach().clone().requires_grad_() for v in aa]
                ea = reference.alpha_composite_batch_selected(*aa, distortions=mode, detach_l1_weights=detached)
                eb = checkpoint(lambda *values:tuple(module.composite(*values, mode, detached)),
                    *bb, use_reentrant=False)
                for x, y in zip(ea, eb):
                    torch.testing.assert_close(x, y, atol=3e-5, rtol=3e-4)
                    maximum = max(maximum, float((x-y).detach().abs().max()))
                la = sum(x.sum() for x in ea if x.requires_grad)
                lb = sum(x.sum() for x in eb if x.requires_grad)
                ga = torch.autograd.grad(la, aa, allow_unused=True)
                gb = torch.autograd.grad(lb, bb, allow_unused=True)
                for x, y, p, q in zip(ga, gb, aa, bb):
                    torch.testing.assert_close(torch.zeros_like(p) if x is None else x,
                        torch.zeros_like(q) if y is None else y, atol=3e-4, rtol=3e-3)
    return dict(passed=True, device=device, event_fixtures=2, composite_fixtures=16, checkpointed=True,
        forward_max_abs=maximum, scope="compiled synthetic ATen/adjoint fixtures only")
