"""Optional cross-view abstention for contradictory SAM family identities.

SPDX-License-Identifier: Apache-2.0
This owned input-confidence rule is not an Unbiased Surfel paper equation.
"""
from __future__ import annotations
import torch
from .family_roles import ordered_native_layers
from .regional_commercial import select_opaque_geometry
from .multiview import _sample_hwc, _in_bounds


@torch.no_grad()
def visible_identity_conflicts(reference_layers, neighbor_layers, reference, neighbor, *,
                               depth_tolerance, reprojection_tolerance=1., hair_family='pgsr'):
    """Only disagreeing labels on the same opaque physical surface qualify.

    The original TSDF voxel supplies the depth tolerance. The existing native
    multi-view one-pixel roundtrip guard and four-sample validity are retained.
    Neither labels nor this detached diagnostic change geometry validity.
    """
    if depth_tolerance<=0 or reprojection_tolerance<=0:
        raise ValueError('Identity correspondence tolerances must be positive')
    source=select_opaque_geometry(*ordered_native_layers(reference_layers,hair_family))
    target=select_opaque_geometry(*ordered_native_layers(neighbor_layers,hair_family))
    h,w=source['depth'].shape
    if target['depth'].shape!=(h,w):raise ValueError('Identity views must share image dimensions')
    dtype=source['depth'].dtype;device=source['depth'].device
    yy,xx=torch.meshgrid(torch.arange(h,device=device,dtype=dtype)+.5,
                        torch.arange(w,device=device,dtype=dtype)+.5,indexing='ij')
    pixels=torch.stack((xx,yy),-1).reshape(-1,2)
    rays=torch.cat((pixels,torch.ones_like(pixels[:,:1])),-1)@torch.linalg.inv(reference['K']).T
    camera=rays*source['depth'].reshape(-1,1)
    world=camera@reference['c2w'][:3,:3].T+reference['c2w'][:3,3]
    view=torch.linalg.inv(neighbor['c2w']);other=world@view[:3,:3].T+view[:3,3]
    q=other@neighbor['K'].T;uv=q[:,:2]/q[:,2:].clamp_min(1e-8)
    target_support=target['valid']&neighbor['foreground']&neighbor['known']&~neighbor['unknown']
    fields=torch.stack((target['depth'],target_support.to(dtype),neighbor['hair']),-1)
    sampled=_sample_hwc(fields,uv)
    other_rays=torch.cat((uv,torch.ones_like(uv[:,:1])),-1)@torch.linalg.inv(neighbor['K']).T
    other_surface=other_rays*sampled[:,:1]
    back_world=other_surface@neighbor['c2w'][:3,:3].T+neighbor['c2w'][:3,3]
    ref_view=torch.linalg.inv(reference['c2w']);back=back_world@ref_view[:3,:3].T+ref_view[:3,3]
    back_q=back@reference['K'].T;back_uv=back_q[:,:2]/back_q[:,2:].clamp_min(1e-8)
    support=(source['valid']&reference['foreground']&reference['known']&~reference['unknown']).reshape(-1)
    support&=_in_bounds(uv,width=w,height=h)&(other[:,2]>.01)&(back[:,2]>.01)&(sampled[:,1]>=.999)
    support&=((sampled[:,0]-other[:,2]).abs()<=depth_tolerance)
    support&=(torch.linalg.vector_norm(back_uv-pixels,dim=-1)<reprojection_tolerance)
    hair=reference['hair'].reshape(-1);other_hair=sampled[:,2]
    disagreement=((hair>=.99)&(other_hair<=.01))|((hair<=.01)&(other_hair>=.99))
    return (support&disagreement).reshape(h,w)


class RegionalIdentityEvidence:
    """Retain distinct existing camera-neighbor witnesses, never learned labels."""
    def __init__(self):
        self.witnesses={};self.calls=0

    def observe(self,frame,neighbor_slot,conflicts):
        if not 0<=neighbor_slot<8:raise ValueError('Witness bits require the existing eight-neighbor graph')
        if conflicts.dtype!=torch.bool or conflicts.ndim!=2:raise ValueError('Conflicts must be boolean [H,W]')
        if frame not in self.witnesses:self.witnesses[frame]=torch.zeros_like(conflicts,dtype=torch.uint8)
        if self.witnesses[frame].shape!=conflicts.shape:raise ValueError('Witness image shape changed')
        self.witnesses[frame]|=conflicts.to(torch.uint8)*(1<<neighbor_slot);self.calls+=1

    def apply(self,frame,item):
        if frame not in self.witnesses:return item
        bits=self.witnesses[frame]
        if bits.shape!=item['unknown'].shape:raise ValueError('Witness image shape changed')
        # Two different neighbors must agree that the original label conflicts.
        uncertain=(bits&(bits-1))!=0
        return dict(item,unknown=item['unknown']|uncertain)

    def state_dict(self):
        return dict(calls=self.calls,witnesses={i:v.cpu() for i,v in self.witnesses.items()})

    def load_state_dict(self,state,*,device):
        self.calls=int(state['calls'])
        if self.calls<0:raise ValueError('Invalid evidence count')
        self.witnesses={int(i):v.to(device=device) for i,v in state['witnesses'].items()}
        if any(v.dtype!=torch.uint8 or v.ndim!=2 for v in self.witnesses.values()):raise ValueError('Invalid witness state')

    @property
    def uncertain_pixels(self):
        return sum(int(((v&(v-1))!=0).sum()) for v in self.witnesses.values())
