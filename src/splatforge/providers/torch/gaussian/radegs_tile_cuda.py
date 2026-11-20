"""Explicit native RaDe-GS tile execution, with the eager projector retained.

The event map and selected native compositor form one execution backend. There
is no event pruning, shape padding change, compiler, geometry adapter or fallback.
"""
from pathlib import Path
import os

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

from .cuda_composite import composite

_extension = None
activation_counts = dict(tile_forward=0, event_forward=0, event_backward=0, event_inspection=0,
    live_support=0, support_event_forward=0, support_event_backward=0,
    tensor_primal_forward=0, tensor_primal_no_grad=0, primal_native_pullback=0)


def extension():
    global _extension
    if _extension is None:
        from torch.utils.cpp_extension import load
        root = os.environ.get('TORCH_EXTENSIONS_DIR')
        if not root:
            raise RuntimeError('Set TORCH_EXTENSIONS_DIR before loading the RaDe-GS CUDA extension')
        from ...._paths import output_path

        build = output_path(Path(root) / 'radegs_tile_events')
        build.mkdir(parents=True, exist_ok=True)
        _extension = load(name='sf_radegs_tile_events',
            sources=[str(Path(__file__).with_suffix('.cu'))],
            build_directory=str(build), extra_cflags=['-O3'],
            extra_cuda_cflags=['-O3', '--fmad=false'], verbose=False)
    return _extension


def event_reference(delta, q2, opacity, length, gain, ray_norm, valid, near, far, extent):
    """Independent ordinary tensor equations for bounded CPU qualification."""
    squared = torch.einsum('bnpi,bnij,bnpj->bnp', delta, q2, delta)
    depth = (length[..., None] + (gain[:, :, None] * delta).sum(-1)) / ray_norm[:, None]
    response = torch.exp(-0.5 * squared)
    response = torch.where(valid[..., None] & (squared <= extent**2) & (depth > near) & (depth < far), response, 0)
    return opacity[..., None] * response, depth


def event_pullback(delta, q2, opacity, length, gain, ray_norm, valid, ga, gd, near, far, extent):
    """Analytic tensor control; production executes the corresponding CUDA VJP."""
    a, depth = event_reference(delta, q2, opacity, length, gain, ray_norm, valid, near, far, extent)
    squared = torch.einsum('bnpi,bnij,bnpj->bnp', delta, q2, delta)
    response = torch.where(valid[..., None] & (squared <= extent**2) & (depth > near) & (depth < far), torch.exp(-0.5*squared), 0)
    gq = -0.5 * (ga * opacity[..., None] * response)
    gz = gd / ray_norm[:, None]
    first = torch.einsum('bnij,bnpj->bnpi', q2 + q2.transpose(-1, -2), delta)
    return (gq[..., None] * first + gz[..., None] * gain[:, :, None],
            (gq[..., None, None] * delta[..., :, None] * delta[..., None, :]).sum(2),
            (ga * response).sum(2), gz.sum(2), (gz[..., None] * delta).sum(2))


class _Events(torch.autograd.Function):
    @staticmethod
    def forward(ctx, delta, q2, opacity, length, gain, ray_norm, valid, near, far, extent):
        ctx.set_materialize_grads(False)
        ctx.settings = near, far, extent
        ctx.save_for_backward(delta, q2, opacity, length, gain, ray_norm, valid)
        activation_counts['event_forward'] += 1
        return tuple(extension().forward(delta, q2, opacity, length, gain, ray_norm, valid, near, far, extent))

    @staticmethod
    @once_differentiable
    def backward(ctx, ga, gd):
        delta, q2, opacity, length, gain, ray_norm, valid = ctx.saved_tensors
        shape = delta.shape[:-1]
        ga = delta.new_zeros(shape) if ga is None else ga.contiguous()
        gd = delta.new_zeros(shape) if gd is None else gd.contiguous()
        result = extension().backward(delta, q2, opacity, length, gain, ray_norm, valid, ga, gd, *ctx.settings)
        activation_counts['event_backward'] += 1
        return (*result, None, None, None, None, None)


def events(delta, q2, opacity, length, gain, ray_norm, valid, *, near, far, extent):
    if any(t.device.type != 'cuda' or t.dtype != torch.float32 for t in (delta, q2, opacity, length, gain, ray_norm)):
        raise ValueError('native tile requires explicit CUDA float32; no fallback')
    if valid.device != delta.device or valid.dtype != torch.bool or ray_norm.requires_grad:
        raise ValueError('fixed CUDA boolean membership and constant camera rays required')
    if delta.numel() >= 2**31:
        raise ValueError('native tile requires original bounded event batches')
    return _Events.apply(delta, q2, opacity, length, gain, ray_norm, valid, near, far, extent)


@torch.no_grad()
def live_support(delta, q2, length, gain, ray_norm, valid, *, near, far, extent):
    """Original represented FP32 decision on live, original-shaped operands."""
    squared = torch.einsum('tlpi,tlij,tlpj->tlp', delta, q2, delta)
    depth = (length[:, :, None] + (gain[:, :, None, :] * delta).sum(-1)) / ray_norm[:, None]
    return (valid[:, :, None] & (squared <= extent**2) & (depth > near) & (depth < far)).contiguous()


class _SupportEvents(torch.autograd.Function):
    @staticmethod
    def forward(ctx, delta, q2, opacity, length, gain, ray_norm, valid, support):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(delta, q2, opacity, length, gain, ray_norm, valid, support)
        activation_counts['event_forward'] += 1
        activation_counts['support_event_forward'] += 1
        return tuple(extension().forward_support(delta, q2, opacity, length, gain, ray_norm, valid, support))

    @staticmethod
    @once_differentiable
    def backward(ctx, ga, gd):
        delta, q2, opacity, length, gain, ray_norm, valid, support = ctx.saved_tensors
        shape = delta.shape[:-1]
        ga = delta.new_zeros(shape) if ga is None else ga.contiguous()
        gd = delta.new_zeros(shape) if gd is None else gd.contiguous()
        result = extension().backward_support(delta, q2, opacity, length, gain, ray_norm, valid, ga, gd, support)
        activation_counts['event_backward'] += 1
        activation_counts['support_event_backward'] += 1
        return (*result, None, None, None)


def events_with_live_support(delta, q2, opacity, length, gain, ray_norm, valid, *, near, far, extent):
    if any(t.device.type != 'cuda' or t.dtype != torch.float32 for t in (delta, q2, opacity, length, gain, ray_norm)):
        raise ValueError('live support native events require CUDA float32; no fallback')
    if valid.device != delta.device or valid.dtype != torch.bool or ray_norm.requires_grad:
        raise ValueError('fixed CUDA boolean membership and constant camera rays required')
    if delta.numel() >= 2**31:
        raise ValueError('original bounded event batches required')
    support = live_support(delta, q2, length, gain, ray_norm, valid, near=near, far=far, extent=extent)
    activation_counts['live_support'] += 1
    return _SupportEvents.apply(delta, q2, opacity, length, gain, ray_norm, valid, support)


def inspect_pixel(delta,q2,opacity,length,gain,ray_norm,valid,batch_row,pixel_column,*,near,far,extent):
    """Read-only actual CUDA diagnostic; calls the production quadratic routine."""
    if delta.device.type!='cuda' or delta.dtype!=torch.float32:
        raise ValueError('native inspection requires CUDA float32; no CPU surrogate')
    if not (0<=batch_row<delta.shape[0] and 0<=pixel_column<delta.shape[2]):
        raise ValueError('native inspection pixel outside original batch')
    result=extension().inspect_pixel(delta,q2,opacity,length,gain,ray_norm,valid,
        batch_row,pixel_column,near,far,extent)
    activation_counts['event_inspection']+=1
    return result


def render_tile(delta, q2, color, opacity, means, normal, gain, valid, selection, rays, background, *, near, far, extent, distortions, use_live_support=False):
    activation_counts['tile_forward'] += 1
    event_function = events_with_live_support if use_live_support else events
    a, depth = event_function(delta, q2, opacity, torch.linalg.vector_norm(means, dim=-1), gain,
        torch.linalg.vector_norm(rays, dim=-1), valid, near=near, far=far, extent=extent)
    pixel_normals = normal[:, :, None].expand(-1, -1, rays.shape[1], -1)
    rgb, alpha, znum, nnum, l1, l2, median, weights = composite(a, color, depth, pixel_normals, distortions=distortions)
    expected = torch.where(alpha > 1e-8, znum / alpha.clamp_min(1e-8), 0)
    ambiguity = (weights * selection[..., None].to(weights.dtype)).sum(1)
    return torch.cat((rgb + (1-alpha[..., None])*background, alpha[..., None], expected[..., None],
        F.normalize(nnum, dim=-1), l1[..., None], l2[..., None], median[..., None],
        torch.zeros_like(alpha)[..., None], ambiguity[..., None]), -1)


def render_live_support_tile(*args, **kwargs):
    return render_tile(*args, **kwargs, use_live_support=True)


class _OriginalPrimalNativePullback(torch.autograd.Function):
    @staticmethod
    def forward(ctx, primal, delta, q2, color, opacity, means, normal, gain,
                valid, selection, rays, background, near, far, extent, distortions):
        ctx.set_materialize_grads(False)
        ctx.settings = dict(near=near, far=far, extent=extent, distortions=distortions)
        # Save only live inputs; the original Tensor intermediate graph is not
        # retained. This node owns its per-batch adjoint state.
        ctx.save_for_backward(delta, q2, color, opacity, means, normal, gain,
            valid, selection, rays, background)
        result = primal()
        activation_counts['tensor_primal_forward'] += 1
        activation_counts['tensor_primal_no_grad'] += int(not torch.is_grad_enabled())
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        if gradient is None:
            return (None,)*16
        saved = ctx.saved_tensors
        with torch.enable_grad():
            values = [v.detach().requires_grad_(True) for v in saved[:7]]
            result = render_live_support_tile(*values, *saved[7:], **ctx.settings)
            gradients = torch.autograd.grad(result, values, gradient,
                create_graph=False, retain_graph=False)
        activation_counts['primal_native_pullback'] += 1
        return (None, *gradients, None, None, None, None, None, None, None, None)


def render_original_primal_native_pullback(primal, delta, q2, color, opacity, means,
        normal, gain, valid, selection, rays, background, *, near, far, extent, distortions):
    if any(v.device.type != 'cuda' or v.dtype != torch.float32
           for v in (delta, q2, color, opacity, means, normal, gain, rays, background)):
        raise ValueError('original-primal/native-pullback requires CUDA float32; no fallback')
    return _OriginalPrimalNativePullback.apply(primal, delta, q2, color, opacity, means,
        normal, gain, valid, selection, rays, background, near, far, extent, distortions)
