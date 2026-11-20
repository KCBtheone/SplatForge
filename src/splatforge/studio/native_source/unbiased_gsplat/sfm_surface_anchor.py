"""Experimental surface constraints from ordinary input SfM training tracks.

SPDX-License-Identifier: Apache-2.0
This is an owned additional constraint, not an Unbiased Surfel paper equation.
It never reads an evaluator, reference mesh, manual anchor list or trained seed.
"""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn.functional as F


def surface_anchor_fields(layers, *, domain='native_surfel', hair_family='pgsr'):
    """Choose the measured field without hiding an erroneous opaque front layer.

    The historical native domain retains its original visibility mask. The
    optional scene domain uses the same native opaque selector as geometry
    export; only that selected family's depth receives the measurement VJP.
    """
    if domain == 'native_surfel':
        layer = layers['unbiased']
        valid = layer['valid'] & (layer['alpha'] >= .5)
        if 'geometry_visibility' in layer:
            valid = valid & layer['geometry_visibility']
        return layer['depth'], valid
    if domain != 'opaque_scene':
        raise ValueError('Unknown input-SfM surface domain')
    if set(layers) != {'pgsr', 'unbiased'}:
        raise ValueError('Opaque-scene SfM constraints require simultaneous joint layers')
    from .family_roles import ordered_native_layers
    from .regional_commercial import select_opaque_geometry
    surface = select_opaque_geometry(*ordered_native_layers(layers, hair_family))
    return surface['depth'], surface['valid']


def build_training_targets(inputs, *, minimum_observations=3, minimum_parallax_degrees=15., reprojection_max=2.):
    """Use only original training observations with trusted non-hair identity."""
    if minimum_observations<3 or not (0<minimum_parallax_degrees<180) or reprojection_max<=0:
        raise ValueError('Invalid SfM input-quality requirements')
    lookup={int(pid):i for i,pid in enumerate(inputs.point_ids)};tracks={}
    for frame in inputs.train_indices:
        row=inputs.rows[frame];item=inputs.load(frame);c2w=np.asarray(row['c2w'],dtype=np.float64)
        view=np.linalg.inv(c2w);k=np.asarray(row['K'],dtype=np.float64)
        trusted=(item['foreground']&item['known']&~item['unknown']&(item['body']>=.99)&(item['hair']<.99))
        h,w=trusted.shape
        for record in row['observations']:
            pid=int(record['point']);index=lookup.get(pid)
            if index is None:continue
            point=np.asarray(inputs.points[index],dtype=np.float64)
            camera=view[:3,:3]@point+view[:3,3]
            if not np.isfinite(camera).all() or camera[2]<=.01:continue
            uv=k@camera;uv=uv[:2]/uv[2]
            if np.linalg.norm(uv-np.array([record['x'],record['y']]))>reprojection_max:continue
            x,y=np.floor(uv-.5).astype(int)
            if not (0<=x<w-1 and 0<=y<h-1) or not trusted[y:y+2,x:x+2].all():continue
            ray=point-c2w[:3,3];length=np.linalg.norm(ray)
            if not np.isfinite(length) or length<=0:continue
            tracks.setdefault(pid,[]).append((frame,uv,float(camera[2]),ray/length))
    per_frame=[[] for _ in inputs.rows];kept_points=0
    for records in tracks.values():
        if len(records)<minimum_observations:continue
        rays=np.stack([r[3] for r in records])
        angle=float(np.degrees(np.arccos(np.clip((rays@rays.T).min(),-1.,1.))))
        if angle<minimum_parallax_degrees:continue
        kept_points+=1
        for frame,uv,z,_ in records:per_frame[frame].append((float(uv[0]),float(uv[1]),z))
    targets=[]
    for records in per_frame:
        values=np.asarray(records,dtype=np.float32).reshape(-1,3)
        targets.append((values[:,:2].copy(),values[:,2].copy()))
    return targets,dict(points=kept_points,observations=sum(len(v) for v in per_frame),
        covered_training_frames=sum(bool(per_frame[i]) for i in inputs.train_indices),
        minimum_observations=minimum_observations,minimum_parallax_degrees=minimum_parallax_degrees,
        reprojection_max=reprojection_max,source='Original trusted non-hair SfM training observations only')


def surface_anchor_loss(depth,valid,uv,target_depth,*,scene_radius=1.):
    """Mean squared camera-z residual; four-sample validity and fixed targets.

    Pixel coordinates denote centers, exactly as in the existing PGSR sampling
    operator. Differences beyond scene_radius/4 abstain, matching the existing
    convergence separation bound. No sparse-measurement gradient is optimized.
    """
    if depth.ndim!=2 or valid.shape!=depth.shape or valid.dtype!=torch.bool:
        raise ValueError('Depth and boolean validity must share [H,W]')
    if uv.ndim!=2 or uv.shape[1]!=2 or target_depth.shape!=(len(uv),):
        raise ValueError('Targets must have shapes [N,2] and [N]')
    if any(t.device!=depth.device or t.dtype!=depth.dtype for t in [uv,target_depth]):
        raise ValueError('Measurement tensors must share depth device and dtype')
    if not math.isfinite(scene_radius) or scene_radius<=0:raise ValueError('Invalid scene radius')
    finite=valid&torch.isfinite(depth);storage=torch.where(finite,depth,torch.zeros_like(depth))
    if len(uv)==0:return storage.sum()*0,torch.zeros((),device=depth.device,dtype=torch.int64)
    uv=uv.detach();target_depth=target_depth.detach();h,w=depth.shape
    target_valid=torch.isfinite(uv).all(-1)&torch.isfinite(target_depth)&(target_depth>0)
    target_valid&=(uv[:,0]>=.5)&(uv[:,0]<=w-.5)&(uv[:,1]>=.5)&(uv[:,1]<=h-.5)
    safe_uv=torch.where(torch.isfinite(uv),uv,torch.zeros_like(uv))
    grid=torch.stack((2*safe_uv[:,0]/w-1,2*safe_uv[:,1]/h-1),-1).reshape(1,-1,1,2)
    source=torch.stack((storage,finite.to(depth.dtype)),0)[None]
    sampled=F.grid_sample(source,grid,mode='bilinear',padding_mode='zeros',align_corners=False)[0,:,:,0]
    error=sampled[0]-target_depth
    active=target_valid&(sampled[1]>=.999)&torch.isfinite(error)&(error.abs()<=scene_radius/4)
    safe_error=torch.where(active,error,torch.zeros_like(error));count=active.sum()
    return safe_error.square().sum()/count.clamp_min(1),count
