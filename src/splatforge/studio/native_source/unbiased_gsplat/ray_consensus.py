"""Experimental ray-depth consensus; current trainers do not call this module.

SPDX-License-Identifier: Apache-2.0
The original adjacent convergence remains the production default. This is an
owned engineering constraint, not the Unbiased Surfel paper's equation 7.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import torch
from torch.autograd.function import once_differentiable
from .convergence import asymmetric_squared_difference

KERNEL_SOURCE=Path(__file__).with_name('ray_consensus.cu')
_MODULES={}
NATIVE_CALLS=dict(compile=0,forward=0,backward=0)


class _WeightedConsensus(torch.autograd.Function):
    @staticmethod
    def forward(ctx,depth,weight):
        g=weight.detach();total=g.sum();centered=depth-(g*depth).sum()/total
        ctx.save_for_backward(centered,g,total)
        return 2*(g*centered.square()).sum()

    @staticmethod
    @once_differentiable
    def backward(ctx,upstream):
        centered,g,total=ctx.saved_tensors
        before_g=g.cumsum(0)-g;before_centered=(g*centered).cumsum(0)-g*centered
        result=4*g*centered+g/total*(before_g*centered-before_centered)
        return upstream*result,None


def consensus_reference(depths,weights,*,scene_radius=1.):
    """One already accepted, center-ordered ray; no opacity or target input.

    Gaussian weights, gap and compactness membership are detached. Components
    split at gaps greater than radius/4. Compact components use the weighted
    complete-pair moment; wide components retain the original adjacent formula.
    """
    if depths.ndim!=1 or weights.shape!=depths.shape:
        raise ValueError('Accepted depth and Gaussian weights must share [N]')
    if not depths.is_floating_point() or weights.dtype!=depths.dtype:
        raise TypeError('Depth and weights must share a floating dtype')
    if depths.device!=weights.device:raise ValueError('Depth/weights device mismatch')
    if not float(scene_radius)>0 or not torch.isfinite(torch.tensor(float(scene_radius))):
        raise ValueError('Scene radius must be finite and positive')
    if not bool(torch.isfinite(depths).all()) or not bool(torch.isfinite(weights).all()) or bool((weights<0).any()):
        raise ValueError('Accepted rays require finite depths and nonnegative weights')
    total=depths.sum()*0
    if len(depths)<2:return total
    quarter=float(scene_radius)/4
    breaks=torch.nonzero((depths[1:].detach()-depths[:-1].detach()).abs()>quarter,as_tuple=False).flatten().cpu().tolist()
    edges=[0]+[i+1 for i in breaks]+[len(depths)]
    for first,last in zip(edges,edges[1:]):
        d=depths[first:last];g=weights[first:last].detach()
        if len(d)<2 or not bool(g.sum()>0):continue
        if float(d.detach().max()-d.detach().min())<=quarter:
            total=total+_WeightedConsensus.apply(d,g)
        else:
            total=total+(torch.minimum(g[:-1],g[1:])*asymmetric_squared_difference(d[:-1],d[1:])).sum()
    return total


def native_available():
    return torch.cuda.is_available() and importlib.util.find_spec('cupy') is not None


def _get_kernels(device):
    # Direct NVRTC API: no source hashing or source-file cache/verification.
    # The CUDA source is immutable for the life of this process's module.
    index=device.index if device.index is not None else torch.cuda.current_device()
    if index in _MODULES:return _MODULES[index]
    import cupy
    from cupy.cuda import nvrtc,function
    with cupy.cuda.Device(index):
        major,minor=torch.cuda.get_device_capability(index)
        program=nvrtc.createProgram(KERNEL_SOURCE.read_text(encoding='utf-8'),'ray_consensus.cu',(),())
        try:
            try:nvrtc.compileProgram(program,('--std=c++14',f'--gpu-architecture=compute_{major}{minor}'))
            except Exception as error:raise RuntimeError('Owned ray-consensus NVRTC build failed: '+nvrtc.getProgramLog(program)) from error
            ptx=nvrtc.getPTX(program)
        finally:nvrtc.destroyProgram(program)
        module=function.Module();module.load(ptx)
        kernels=(module,module.get_function('ray_consensus_forward'),module.get_function('ray_consensus_backward'))
    _MODULES[index]=kernels;NATIVE_CALLS['compile']+=1
    return kernels


def _launch(function,tensors,scalars,pixels,device):
    import cupy
    import numpy as np
    stream=torch.cuda.current_stream(device)
    with cupy.cuda.Device(device),cupy.cuda.ExternalStream(stream.cuda_stream):
        arrays=tuple(cupy.from_dlpack(t.detach()) for t in tensors)
        function(((pixels+127)//128,),(128,),arrays+tuple(np.int32(v) if isinstance(v,int) else np.float32(v) for v in scalars),stream=cupy.cuda.get_current_stream())


class _NativeConsensus(torch.autograd.Function):
    @staticmethod
    def forward(ctx,means,ray,opacity,offsets,ids,width,height,tile,quarter):
        module,fwd,bwd=_get_kernels(ray.device)
        loss=torch.empty((height,width),device=ray.device,dtype=ray.dtype)
        compact=torch.empty((height,width),device=ray.device,dtype=torch.int32)
        wide=torch.empty_like(compact)
        _launch(fwd,(means,ray,opacity,offsets,ids,loss,compact,wide),
                (ray.shape[1],ids.numel(),width,height,tile,quarter),width*height,ray.device.index)
        NATIVE_CALLS['forward']+=1
        ctx.save_for_backward(means,ray,opacity,offsets,ids)
        ctx.geometry=(width,height,tile,quarter);ctx.kernel=(module,bwd)
        ctx.mark_non_differentiable(compact,wide)
        return loss,compact,wide

    @staticmethod
    @once_differentiable
    def backward(ctx,grad_loss,grad_compact,grad_wide):
        means,ray,opacity,offsets,ids=ctx.saved_tensors;width,height,tile,quarter=ctx.geometry
        gradient=torch.zeros_like(ray)
        if grad_loss is not None:
            _launch(ctx.kernel[1],(means,ray,opacity,offsets,ids,grad_loss.contiguous(),gradient),
                    (ray.shape[1],ids.numel(),width,height,tile,quarter),width*height,ray.device.index)
            NATIVE_CALLS['backward']+=1
        return None,gradient,None,None,None,None,None,None,None


def consensus_from_gsplat(meta,width,height,*,scene_radius=1.):
    """Dense one-camera pinned-gsplat metadata; no packed-mode index guessing.

    Returns an unnormalized dense loss map and compact/wide component counters.
    No Torch fallback is selected when this explicit native operator is called.
    """
    if not isinstance(width,int) or not isinstance(height,int) or width<=0 or height<=0:
        raise ValueError('Image dimensions must be positive integers')
    if not float(scene_radius)>0 or not torch.isfinite(torch.tensor(float(scene_radius))):
        raise ValueError('Scene radius must be finite and positive')
    ray=meta['ray_transforms'];means=meta['means2d'];opacity=meta['opacities']
    if ray.ndim!=4 or ray.shape[0]!=1 or ray.shape[-2:]!=(3,3):raise ValueError('Only dense one-camera [1,N,3,3] ray transforms are supported')
    n=ray.shape[1]
    if means.shape!=(1,n,2) or opacity.shape!=(1,n):raise ValueError('Dense metadata shape mismatch')
    if meta.get('gaussian_ids') is not None or meta.get('camera_ids') is not None:raise ValueError('Packed metadata is not supported')
    if ray.dtype!=torch.float32 or means.dtype!=ray.dtype or opacity.dtype!=ray.dtype:raise TypeError('Native consensus requires float32 metadata')
    if not ray.is_cuda or means.device!=ray.device or opacity.device!=ray.device:raise ValueError('Native consensus requires one shared CUDA device')
    if not native_available():raise RuntimeError('Explicit native ray-consensus backend is unavailable')
    tile=int(meta['tile_size']);offsets=meta['isect_offsets'];ids=meta['flatten_ids']
    if tile<=0 or offsets.numel()!=((width+tile-1)//tile)*((height+tile-1)//tile):raise ValueError('Tile-offset geometry mismatch')
    if offsets.device!=ray.device or ids.device!=ray.device:raise ValueError('Tile metadata must share the ray CUDA device')
    if n>=2**31 or ids.numel()>=2**31:raise ValueError('Dense metadata exceeds int32 index range')
    return _NativeConsensus.apply(means.contiguous(),ray.contiguous(),opacity.contiguous(),offsets.to(torch.int32).contiguous(),ids.to(torch.int32).contiguous(),width,height,tile,float(scene_radius)/4)
