"""Owned candidate-order geometry, independent of photometric composition.

SPDX-License-Identifier: Apache-2.0
Experimental: native numeric qualification is required before training adoption.
"""
from __future__ import annotations
import inspect
from types import FunctionType
from functools import update_wrapper
import torch
from .physical_ordering import physical_ordered_render, load_physical_order_native, PhysicalOrderedOutputs

CANDIDATE_GEOMETRY_FORWARD_CALLS = 0

def geometry_layers(layers):
    """Explicit consumer view; original RGB/alpha/sort tensors stay in layers."""
    return {name: layer.get('geometry', layer) for name, layer in layers.items()}


def dual_geometry_rasterizer(original, **geometry_options):
    """Capture permitted gsplat projected fields without replacing photo output."""
    callback = original.__globals__.get('rasterize_to_pixels_2dgs')
    if not callable(callback):
        raise RuntimeError('Dual geometry requires the permitted gsplat pixel interface')
    signature = inspect.signature(callback)
    state = {}
    def pixels(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        fields = bound.arguments
        if state:
            raise RuntimeError('Dual geometry expected one primary pixel invocation')
        if any(fields.get(k, False) for k in ('packed', 'absgrad', 'distloss')) or fields.get('masks') is not None:
            raise ValueError('Dual geometry supports ordinary unpacked unmasked rendering')
        photo = callback(*args, **kwargs)
        meta = dict(means2d=fields['means2d'], ray_transforms=fields['ray_transforms'],
            opacities=fields['opacities'], gradient_2dgs=fields['densify'],
            isect_offsets=fields['isect_offsets'], flatten_ids=fields['flatten_ids'],
            width=fields['image_width'], height=fields['image_height'], tile_size=fields['tile_size'])
        state['geometry'] = candidate_geometry(meta, fields['normals'].reshape(-1, 3), **geometry_options)
        return photo
    namespace = dict(original.__globals__, rasterize_to_pixels_2dgs=pixels)
    scoped = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
    update_wrapper(scoped, original)
    scoped.__kwdefaults__ = original.__kwdefaults__
    return scoped, state


class CandidateGeometryNative:
    """Reuse owned projective/compositing math with original candidate order.

    Only raw-normal weight adjoints enter the independent screen statistic.
    All field cotangents still enter the real physical parameter VJP.
    """
    def __init__(self, native):
        self.native = native

    def forward(self, *args):
        global CANDIDATE_GEOMETRY_FORWARD_CALLS
        result = self.native.forward(*args, True)
        CANDIDATE_GEOMETRY_FORWARD_CALLS += 1
        return result

    def backward(self, *args):
        result = list(self.native.backward(*args, True))
        normal_only = list(args)
        # ABI: twelve saved tensors, RGB/alpha/normal/depth/convergence, constants.
        for index in (12, 13, 15, 16):
            normal_only[index] = torch.zeros_like(args[index])
        result[5] = self.native.backward(*normal_only, True)[5]
        return tuple(result)


def candidate_geometry(meta, normals_camera, *, midpoint, terminal, front, quarter):
    """Consume exactly the primary projection's unreordered tile candidates."""
    means = meta['means2d']
    if means.ndim != 3 or means.shape[0] != 1:
        raise ValueError('Dual geometry requires one unpacked camera')
    width, height, tile = (int(meta[k]) for k in ('width', 'height', 'tile_size'))
    rays = meta['ray_transforms'].reshape(-1, 3, 3).contiguous()
    screen = meta['gradient_2dgs'].reshape(-1, 2)
    return physical_ordered_render(means.reshape(-1, 2).contiguous(), rays,
        torch.zeros_like(normals_camera), meta['opacities'].reshape(-1).contiguous(),
        normals_camera.contiguous(),
        meta['isect_offsets'].reshape((height+tile-1)//tile, (width+tile-1)//tile).to(torch.int32).contiguous(),
        meta['flatten_ids'].reshape(-1).to(torch.int32).contiguous(),
        torch.arange(width*height, device=means.device, dtype=torch.int64),
        width, height, tile, midpoint=midpoint, terminal=terminal, front=front,
        quarter=quarter, screen=screen,
        native=CandidateGeometryNative(load_physical_order_native()))


def candidate_geometry_reference(means, rays, colors, opacity, normals, offsets,
                                 ids, pixels, width, height, tile, *, midpoint=True,
                                 terminal=True, front=True, quarter=1.):
    """Tiny owned tensor specification for neutral fixtures, never a fallback.

    Forward physics and the prescribed slot-dependent surrogate are separate.
    Original CSR slot positions, including all rejected slots, define batches.
    """
    if ids.numel()*pixels.numel()>1000000:
        raise ValueError('Candidate reference is restricted to small numeric fixtures')
    outputs=[[] for _ in range(7)]
    flat_offsets=offsets.reshape(-1)
    for pix in pixels.detach().cpu().tolist():
        x,y=pix%width+.5,pix//width+.5
        t=(pix//width//tile)*((width+tile-1)//tile)+(pix%width//tile)
        lo=int(flat_offsets[t]);hi=int(flat_offsets[t+1]) if t+1<offsets.numel() else ids.numel()
        zero=(means.sum()+rays.sum()+opacity.sum()+normals.sum()+colors.sum())*0
        rgb=colors.new_zeros(3)+zero;normal=normals.new_zeros(3)+zero
        depth=zero;conv=zero;T=zero+1;accum=zero.detach();have=False;selected=-1
        prior=None;anchor=None
        for slot in range(lo,hi):
            g=int(ids[slot]);h=rays[g].double();m=means[g].double()
            cross=torch.linalg.cross(x*h[2]-h[0],y*h[2]-h[1],dim=-1)
            if not bool(cross[2].detach()!=0):continue
            uv=cross[:2]/cross[2];rho3=uv.square().sum();rho2=2*(m-m.new_tensor([x,y])).square().sum()
            G=(-.5*torch.minimum(rho3,rho2)).exp().float()
            d=torch.where(rho3<=rho2,(uv*h[2,:2]).sum()+h[2,2],h[2,2]).float()
            raw_alpha=opacity[g]*G
            # Prescribed surrogate: capped value, unit adjoint to raw_alpha.
            # Subtract identical raw values before adding the forward cap.
            a=raw_alpha.clamp(max=.99).detach()+(raw_alpha-raw_alpha.detach())
            if not bool(torch.isfinite(d.detach())&torch.isfinite(G.detach())&(d.detach()>=.2)&(a.detach()>=1/255)):continue
            next_T=T*(1-a)
            if bool(next_T.detach()<1e-4):break
            weight=T*a;rgb=rgb+weight*colors[g];normal=normal+weight*normals[g]
            accum=accum+((opacity[g].detach()+.1)*G.detach()).clamp(max=.99)
            crossed=bool(accum>=.6)
            if not have and (crossed or terminal):
                value=.5*(d+prior[1]) if midpoint and prior is not None else d
                depth=d+(value-d).detach();selected=slot-lo;have=crossed
            if prior is not None:
                ps,pd,pG=prior;delta=d-pd
                if (not front or bool(T.detach()>.09)) and bool(delta.detach().abs()<=quarter):
                    common=2*delta.detach()*torch.minimum(G.detach(),pG.detach())
                    current_gain=1.25 if not front or bool(delta.detach()>=0) else 1.
                    prior_gain=1. if not front or bool(delta.detach()>=0) else 1.25
                    surrogate=zero
                    if (hi-1-slot)//256==(hi-1-ps)//256:surrogate=surrogate+current_gain*common*d
                    if anchor is not None and (hi-1-anchor)//256==(hi-1-ps)//256:
                        surrogate=surrogate-prior_gain*common*pd
                    energy=torch.minimum(G.detach(),pG.detach())*delta.detach().square()
                    conv=conv+energy+(surrogate-surrogate.detach())
            anchor=None if prior is None else prior[0]
            prior=(slot,d,G);T=next_T
        for out,value in zip(outputs,(rgb,1-T,normal,depth,have or (terminal and prior is not None),conv,selected)):
            out.append(value)
    return PhysicalOrderedOutputs(*(torch.stack(v) if i not in (4,6) else torch.tensor(v,device=means.device,
        dtype=torch.bool if i==4 else torch.int32) for i,v in enumerate(outputs)))
