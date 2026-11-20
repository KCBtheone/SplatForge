"""Opt-in physically ordered Surfel image formation, initially a prototype.

SPDX-License-Identifier: Apache-2.0
Projection/SH evaluation remain the permitted gsplat caller's responsibility.
The same uncapped per-query ray sequence forms primary RGB/alpha/raw normals
and the existing corrected-depth/front-convergence auxiliaries.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache, update_wrapper
from types import FunctionType
import inspect
import os
from pathlib import Path
import torch


PHYSICAL_PRIMARY_FORWARD_CALLS=0


@dataclass(frozen=True)
class PhysicalOrderedOutputs:
    rgb: torch.Tensor
    alpha: torch.Tensor
    normals: torch.Tensor
    depth: torch.Tensor
    valid: torch.Tensor
    convergence: torch.Tensor
    selected: torch.Tensor


def validate_ordering_inputs(means, rays, colors, opacity, normals, offsets,
                             ids, pixels, width, height, tile):
    if any(type(v) is not int or v <= 0 for v in (width, height, tile)):
        raise ValueError('Physical ordering requires positive integer image/tile dimensions')
    n = means.shape[0] if means.ndim == 2 else -1
    if (n < 0 or means.shape != (n, 2) or rays.shape != (n, 3, 3)
            or colors.shape != (n, 3) or normals.shape != (n, 3)
            or opacity.shape != (n,)):
        raise ValueError('Physical ordering requires single-camera RGB3 projected fields')
    if offsets.shape != ((height+tile-1)//tile, (width+tile-1)//tile):
        raise ValueError('Physical ordering tile-offset shape mismatch')
    if ids.ndim != 1 or pixels.ndim != 1:
        raise ValueError('Physical ordering IDs and query pixels must be vectors')
    fields = (means, rays, colors, opacity, normals, offsets, ids, pixels)
    if any(v.device != means.device or not v.is_contiguous() for v in fields):
        raise ValueError('Physical ordering requires contiguous fields on one device')
    if any(v.dtype != torch.float32 for v in fields[:5]):
        raise ValueError('Physical ordering native fields require float32')
    if offsets.dtype != torch.int32 or ids.dtype != torch.int32 or pixels.dtype != torch.int64:
        raise ValueError('Physical ordering needs int32 tile metadata and int64 query pixels')
    if width*height >= 2**31 or n >= 2**31 or ids.numel() >= 2**31 or pixels.numel() >= 2**31:
        raise ValueError('Physical ordering dimensions exceed native integer indexing; no truncation permitted')
    if pixels.numel() and (bool((pixels < 0).any()) or bool((pixels >= width*height).any())):
        raise ValueError('Physical ordering query pixel outside image')


@lru_cache(maxsize=1)
def load_physical_order_native():
    from torch.utils.cpp_extension import load
    cache = os.environ.get('TORCH_EXTENSIONS_DIR')
    if not cache:
        raise RuntimeError('Physical-order build requires a data-disk TORCH_EXTENSIONS_DIR')
    source = Path(__file__).resolve().parent
    build = Path(cache)/'physical_ordering_native'
    build.mkdir(parents=True, exist_ok=True)
    return load(name='forge_physical_ordering',
                sources=[str(source/'physical_ordering.cpp'), str(source/'physical_ordering.cu')],
                extra_cflags=['-O3'], extra_cuda_cflags=['-O3', '-lineinfo', '-use_fast_math'],
                build_directory=str(build), verbose=False)


class _PhysicalOrder(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means, rays, colors, opacity, normals, screen, offsets, ids, pixels,
                width, height, tile, midpoint, terminal, front, quarter, native):
        global PHYSICAL_PRIMARY_FORWARD_CALLS
        result = native.forward(means, rays, colors, opacity, normals, offsets,
                                ids, pixels, width, height, tile, midpoint,
                                terminal, front, quarter)
        if len(result) != 12:
            raise RuntimeError('Physical-order native forward ABI mismatch')
        PHYSICAL_PRIMARY_FORWARD_CALLS+=1
        rgb, alpha, raw_normals, depth, valid, conv, selected, last, keys, values, prefix, final_T = result
        ctx.save_for_backward(means, rays, colors, opacity, normals, pixels,
                              keys, values, prefix, last, selected, final_T)
        ctx.native = native
        ctx.constants = (width, front, quarter)
        ctx.mark_non_differentiable(valid, selected)
        return rgb, alpha, raw_normals, depth, valid, conv, selected

    @staticmethod
    def backward(ctx, grgb, galpha, gnormal, gdepth, gvalid, gconv, gselected):
        means, rays, colors, opacity, normals, pixels, keys, values, prefix, last, selected, final_T = ctx.saved_tensors
        p = pixels.numel()
        def incoming(value, channels=None):
            shape = (p,) if channels is None else (p, channels)
            return means.new_zeros(shape) if value is None else value.contiguous()
        result = ctx.native.backward(means, rays, colors, opacity, normals,
            pixels, keys, values, prefix, last, selected, final_T,
            incoming(grgb, 3), incoming(galpha), incoming(gnormal, 3),
            incoming(gdepth), incoming(gconv), *ctx.constants)
        if len(result) != 6:
            raise RuntimeError('Physical-order native backward ABI mismatch')
        # Sixth output is the completed primary-only physical screen demand.
        return (*result,) + (None,)*11


def physical_ordered_render(means, rays, colors, opacity, normals, offsets, ids,
                            pixels, width, height, tile, *, midpoint=True,
                            terminal=True, front=True, quarter=1., screen=None, native=None):
    validate_ordering_inputs(means, rays, colors, opacity, normals, offsets,
                             ids, pixels, width, height, tile)
    if not means.is_cuda:
        raise ValueError('Physical-order native renderer requires CUDA; use reference for CPU fixtures')
    if any(type(v) is not bool for v in (midpoint, terminal, front)):
        raise TypeError('Physical-order switches must be boolean')
    if not 0 < float(quarter) < float('inf'):
        raise ValueError('Physical-order convergence gap must be finite and positive')
    if native is None:
        native = load_physical_order_native()
    if screen is None:
        screen = torch.zeros_like(means, requires_grad=means.requires_grad)
    if screen.shape != means.shape or screen.device != means.device or screen.dtype != means.dtype:
        raise ValueError('Physical-order screen-gradient dummy must match projected means')
    result = _PhysicalOrder.apply(means, rays, colors, opacity, normals,
        screen, offsets, ids, pixels, width, height, tile, midpoint, terminal, front,
        float(quarter), native)
    return PhysicalOrderedOutputs(*result)


def physical_primary_rasterizer(original, *, midpoint, terminal, front, quarter):
    """Scope only the permitted renderer's pixel callback, retaining projection/SH.

    Returned state belongs to this ONE render invocation. No global patch or
    last-render cache is used. The existing cached-camera adapter can wrap the
    returned FunctionType without losing its own immutable-inverse contract.
    """
    if not isinstance(original,FunctionType) or original.__globals__.get('torch') is not torch:
        raise RuntimeError('Physical primary needs the permitted gsplat Python rasterizer')
    pixel_original=original.__globals__.get('rasterize_to_pixels_2dgs')
    if not callable(pixel_original) or 'rasterize_to_pixels_2dgs' not in original.__code__.co_names:
        raise RuntimeError('Physical primary pixel callback contract unavailable')
    signature=inspect.signature(pixel_original);state={}
    def pixels(*args,**kwargs):
        bound=signature.bind(*args,**kwargs);bound.apply_defaults();values=bound.arguments
        if (values.get('packed',False) or values.get('absgrad',False) or values.get('distloss',False)
                or values.get('backgrounds') is not None or values.get('masks') is not None):
            raise ValueError('Physical primary requires unpacked RGB without native background/mask/absgrad/distloss')
        if state:raise RuntimeError('Physical primary expected exactly one pixel invocation')
        means=values['means2d'];width=values['image_width'];height=values['image_height'];tile=values['tile_size']
        if means.ndim!=3 or means.shape[0]!=1:
            raise ValueError('Physical primary requires exactly one unpacked camera')
        fields=[values[key].reshape(-1,*shape).contiguous() for key,shape in
            [('means2d',(2,)),('ray_transforms',(3,3)),('colors',(3,)),('opacities',()),('normals',(3,))]]
        result=physical_ordered_render(*fields,
            values['isect_offsets'].reshape((height+tile-1)//tile,(width+tile-1)//tile).to(torch.int32).contiguous(),
            values['flatten_ids'].reshape(-1).to(torch.int32).contiguous(),
            torch.arange(width*height,device=means.device,dtype=torch.int64),width,height,tile,
            midpoint=midpoint,terminal=terminal,front=front,quarter=quarter,
            screen=values['densify'].reshape(-1,2))
        state['ordered']=result
        rgb=result.rgb.reshape(1,height,width,3);alpha=result.alpha.reshape(1,height,width,1)
        normals=result.normals.reshape(1,height,width,3)
        unused=result.depth.new_zeros((1,height,width,1))
        return rgb,alpha,normals,unused,result.depth.reshape(1,height,width,1)
    namespace=dict(original.__globals__);namespace['rasterize_to_pixels_2dgs']=pixels
    scoped=FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__)
    update_wrapper(scoped,original);scoped.__kwdefaults__=original.__kwdefaults__
    return scoped,state


def physical_order_reference(means, rays, colors, opacity, normals, offsets,
                             ids, pixels, width, height, tile, *, midpoint=True,
                             terminal=True, front=True, quarter=1., sequence=None):
    """Independent small-fixture autograd math; never an image-scale fallback.

    Optional sequence fixes native order/admission ONLY for real-field VJP
    comparisons. Independent tiny fixtures qualify sorting and admission too.
    Geometry keeps the original current-only midpoint VJP, first anchor,
    detached-G convergence and deeper-endpoint factor1.25.
    """
    rgb_out=[];alpha_out=[];normal_out=[];depth_out=[];valid_out=[];conv_out=[];selected_out=[]
    tw=(width+tile-1)//tile
    for q,pixel in enumerate(pixels.detach().cpu().tolist()):
        x=pixel%width+.5;y=pixel//width+.5;t=(int(y)//tile)*tw+int(x)//tile
        lo=int(offsets.reshape(-1)[t]);hi=int(offsets.reshape(-1)[t+1]) if t+1<offsets.numel() else ids.numel()
        gid=ids[lo:hi].long() if sequence is None else sequence[q].to(device=means.device,dtype=torch.long)
        h=rays[gid];m=means[gid]
        hu=x*h[:,2]-h[:,0];hv=y*h[:,2]-h[:,1]
        cross=torch.linalg.cross(hu,hv,dim=-1);safe=torch.where(cross[:,2]!=0,cross[:,2],1.)
        uv=cross[:,:2]/safe[:,None]
        rho3=uv.square().sum(-1);rho2=2*(m-m.new_tensor([x,y])).square().sum(-1)
        G=(-.5*torch.minimum(rho3,rho2)).exp();a=(opacity[gid]*G).clamp(max=.999)
        d=torch.where(rho3<=rho2,(uv*h[:,2,:2]).sum(-1)+h[:,2,2],h[:,2,2])
        if sequence is None:
            keep=(cross[:,2]!=0)&torch.isfinite(G)&torch.isfinite(d)&(a.detach()>=1/255)
            order=torch.argsort(d.detach()[keep],stable=True)
            select=torch.nonzero(keep,as_tuple=False).flatten()[order]
            post=(1-a.detach()[select]).cumprod(0)
            select=select[post>1e-4]
            gid=gid[select];G=G[select];a=a[select];d=d[select]
        one=means.new_tensor(1.);zero=means.sum()*0
        T=one;rgb=colors.new_zeros(3)+zero;n=normals.new_zeros(3)+zero
        dep=zero;conv=zero;cumulative=zero.detach();have=False;sel=-1
        previous_d=None;previous_G=None;previous_anchored=False
        for j in range(gid.numel()):
            g=gid[j];weight=T*a[j];rgb=rgb+weight*colors[g];n=n+weight*normals[g]
            cumulative=cumulative+(opacity[g].detach()+.1)*G[j].detach()
            crossed=bool(cumulative>.6)
            if not have and (crossed or terminal):
                value=.5*(d[j]+previous_d) if midpoint and previous_d is not None else d[j]
                dep=d[j]+(value-d[j]).detach();sel=int(g)
                have=crossed
            if previous_d is not None:
                delta=d[j]-previous_d
                if (not front or bool(T.detach()>.09)) and bool(delta.detach().abs()<=quarter):
                    w=torch.minimum(G[j].detach(),previous_G.detach())
                    energy=w*delta.square()
                    near=previous_d if not front or previous_anchored else previous_d.detach()
                    current_scale=1.25 if not front or bool(delta.detach()>=0) else 1.
                    prior_scale=1. if not front or bool(delta.detach()>=0) else 1.25
                    sd=current_scale*d[j]-prior_scale*near
                    surrogate=w*2*delta.detach()*sd
                    conv=conv+energy.detach()+surrogate-surrogate.detach()
            previous_anchored=previous_d is not None
            previous_d=d[j];previous_G=G[j];T=T*(1-a[j])
        rgb_out.append(rgb);alpha_out.append(1-T);normal_out.append(n);depth_out.append(dep)
        valid_out.append(have or (terminal and gid.numel()>0));conv_out.append(conv);selected_out.append(sel)
    if not pixels.numel():
        zero=means.sum()*0
        return PhysicalOrderedOutputs(colors.new_zeros((0,3))+zero,means.new_zeros(0)+zero,
            normals.new_zeros((0,3))+zero,means.new_zeros(0)+zero,
            torch.zeros(0,device=means.device,dtype=torch.bool),means.new_zeros(0)+zero,
            torch.zeros(0,device=means.device,dtype=torch.int32))
    return PhysicalOrderedOutputs(torch.stack(rgb_out),torch.stack(alpha_out),torch.stack(normal_out),
        torch.stack(depth_out),torch.tensor(valid_out,device=means.device),torch.stack(conv_out),
        torch.tensor(selected_out,device=means.device,dtype=torch.int32))
