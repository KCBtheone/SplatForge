"""GOF event evaluation and its first-order float32 CUDA VJP.

The Torch caller supplies the support mask and separate ray and exponent matrices.
"""
import os
from pathlib import Path

import torch
from torch.autograd.function import once_differentiable

from ...._paths import output_path

_extension = None
activation_counts = {'direct_forward': 0, 'direct_backward': 0,
                     'stage_forward':0,'stage_backward':0}


def extension():
    global _extension
    if _extension is None:
        from torch.utils.cpp_extension import load
        root = os.environ.get('TORCH_EXTENSIONS_DIR')
        if not root:
            raise RuntimeError('Set TORCH_EXTENSIONS_DIR before loading the GOF CUDA extension')
        build = output_path(Path(root) / 'gof_direct_h002')
        build.mkdir(parents=True, exist_ok=True)
        _extension = load(name='sf_gof_direct_h002',
                          sources=[str(Path(__file__).with_suffix('.cu'))],
                          build_directory=str(build), extra_cflags=['-O3'],
                          extra_cuda_cflags=['-O3', '--fmad=false'], verbose=False)
    return _extension


class DepthOffset(torch.autograd.Function):
    @staticmethod
    def forward(ctx,proxy,qray,rays):
        if any(x.device.type!='cuda' or x.dtype!=torch.float32 or x.device!=proxy.device
               for x in (proxy,qray,rays)):
            raise ValueError('explicit matching CUDA float32 stage operands required')
        if qray.shape!=proxy.shape or rays.shape!=(proxy.shape[0],proxy.shape[2],3):
            raise ValueError('invalid depth-offset stage shapes')
        result=extension().depth_offset_forward(proxy,qray,rays)
        activation_counts['stage_forward']+=1
        ctx.save_for_backward(proxy,qray,rays,result[0])
        return tuple(result)

    @staticmethod
    @once_differentiable
    def backward(ctx,gd,go):
        result=extension().depth_offset_backward(*ctx.saved_tensors,gd,go)
        activation_counts['stage_backward']+=1
        return tuple(result)


class DirectEvents(torch.autograd.Function):
    @staticmethod
    def forward(ctx, proxy, qr, qe, opacity, rays, squared, valid, near, far, extent):
        activation_counts['direct_forward'] += 1
        ctx.save_for_backward(proxy, qr, qe, opacity, rays, squared, valid)
        ctx.settings = near, far, extent
        return tuple(extension().forward(proxy, qr, qe, opacity, rays, squared, valid, *ctx.settings))

    @staticmethod
    @once_differentiable
    def backward(ctx, alpha, depth, normal):
        activation_counts['direct_backward'] += 1
        return (*extension().backward(*ctx.saved_tensors, *ctx.settings, alpha, depth, normal),
                None, None, None, None, None)


def direct_events(proxy, qr, qe, opacity, rays, squared, valid, *, near, far, extent):
    operands = proxy, qr, qe, opacity, rays, squared
    if any(x.device != proxy.device or x.device.type != 'cuda' or x.dtype != torch.float32
           for x in operands):
        raise ValueError('explicit matching CUDA float32 operands required')
    b, l, p, d = proxy.shape
    if (d != 3 or qr.shape != (b, l, 3, 3) or qe.shape != qr.shape
            or opacity.shape != (b, l) or rays.shape != (b, p, 3)
            or squared.shape != (b, l, p) or valid.shape != (b, l)
            or valid.dtype != torch.bool or valid.device != proxy.device):
        raise ValueError('invalid GOF direct event shapes')
    return DirectEvents.apply(*operands, valid, near, far, extent)
