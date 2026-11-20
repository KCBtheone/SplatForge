"""Independent parameters, refinement and Unbiased rendering.

SPDX-License-Identifier: Apache-2.0
Refinement is a disclosed gsplat-style policy, not an author-trace reproduction.
"""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn.functional as F
from .planar_gaussian import quaternion_rotation
from .trainer_adapter import UnbiasedSurfelConfig, compute_from_gsplat_outputs


def unbiased_regularizer_terms(convergence_loss,normal_loss,*,normal_weight=.05,convergence_weight=7.):
    """Return separate terms so the trainer preserves legacy addition order."""
    if not math.isfinite(normal_weight) or normal_weight<0:
        raise ValueError('Unbiased normal weight must be finite and nonnegative')
    if not math.isfinite(convergence_weight) or convergence_weight<0:
        raise ValueError('Unbiased convergence weight must be finite and nonnegative')
    if convergence_loss.ndim!=0 or normal_loss.ndim!=0:
        raise ValueError('Native Unbiased geometry losses must be scalar')
    return convergence_weight*convergence_loss,normal_weight*normal_loss


def initialize_parameters(points, colors, *, axes, device='cuda', seed=42, orientation_init='uniform_sphere', orientation_report=None):
    if orientation_init not in ('uniform_sphere','positive_uniform','sfm_plane'):
        raise ValueError('Unknown Surfel orientation initialization')
    if axes!=2 and orientation_init!='uniform_sphere':
        raise ValueError('Alternative orientation initialization is Surfel-only')
    from scipy.spatial import cKDTree
    distances,_=cKDTree(points).query(points,k=min(4,len(points)))
    scale=np.sqrt(np.mean(np.square(distances[:,1:]),axis=1)).clip(1e-5,.1)
    generator=torch.Generator(device=device).manual_seed(seed)
    sample_orientation=torch.rand if orientation_init=='positive_uniform' else torch.randn
    n=len(points);sh=torch.zeros(n,16,3,device=device)
    sh[:,0]=(torch.as_tensor(colors,device=device)-.5)/.28209479177387814
    values=dict(means=torch.as_tensor(points,device=device),
                scales=torch.as_tensor(np.log(scale)[:,None].repeat(axes,axis=1),device=device,dtype=torch.float32),
                quats=F.normalize(sample_orientation(n,4,generator=generator,device=device),dim=-1),
                opacities=torch.full((n,),math.log(.1/.9),device=device),
                sh0=sh[:,:1].contiguous(),shN=sh[:,1:].contiguous())
    if orientation_init == 'sfm_plane':
        from .sfm_orientation import sfm_plane_quaternions
        values['quats'], report = sfm_plane_quaternions(points, values['quats'])
        if orientation_report is not None:orientation_report.update(report)
    return {k:torch.nn.Parameter(v.float()) for k,v in values.items()}


def make_optimizer(families, exposure=None):
    rates=dict(means=1.6e-4,scales=5e-3,quats=1e-3,opacities=5e-2,sh0=2.5e-3,shN=1.25e-4)
    groups=[dict(params=[value],lr=rates[key],name=f'{family}:{key}') for family,params in families.items() for key,value in params.items()]
    if exposure is not None:groups.append(dict(params=[exposure],lr=.001,name='exposure'))
    return torch.optim.Adam(groups,eps=1e-15)


def update_mean_learning_rate(optimizer,step,total=30000):
    value=1.6e-4*(.01**(step/total))
    for group in optimizer.param_groups:
        if group['name'].endswith(':means'):group['lr']=value


@torch.no_grad()
def replace_population(parameters,optimizer,family,parents,children,position_offsets=None,scale_shrink=None):
    """Keep retained moments and zero only new children's moments."""
    for key,old in list(parameters.items()):
        values=old.detach()[parents].clone()
        if key=='means' and position_offsets is not None:values+=position_offsets
        if key=='scales' and scale_shrink is not None:values-=scale_shrink[:,None]
        new=torch.nn.Parameter(values);old_state=optimizer.state.pop(old,{})
        state={}
        for name,value in old_state.items():
            if isinstance(value,torch.Tensor) and value.shape==old.shape:
                copied=value[parents].clone();copied[children]=0;state[name]=copied
            else:state[name]=value
        optimizer.state[new]=state
        for group in optimizer.param_groups:
            if group['name']==f'{family}:{key}':group['params']=[new]
        parameters[key]=new


@torch.no_grad()
def refine_population(parameters,optimizer,family,gradient_sum,observation_count,*,max_points=0,
                      reset_split_parent_moments=False,growth_priority_mask=None,surfel_split_scale=.01):
    if type(max_points) is not int or max_points<0:
        raise ValueError('max_points must be a nonnegative integer; 0 means no population cap')
    n=len(parameters['means']);opacity=parameters['opacities'].sigmoid();physical=parameters['scales'].exp()
    if not math.isfinite(surfel_split_scale) or surfel_split_scale<=0:
        raise ValueError('Surfel split scale must be finite and positive')
    if physical.shape[-1]!=2 and surfel_split_scale!=.01:
        raise ValueError('Surfel split scale is only configurable for two-axis surfels')
    if growth_priority_mask is not None:
        if growth_priority_mask.dtype!=torch.bool or growth_priority_mask.shape!=(n,) or growth_priority_mask.device!=parameters['means'].device:
            raise ValueError('Growth priority must be a boolean mask matching the current population and device')
    finite=torch.isfinite(parameters['means']).all(-1)&torch.isfinite(physical).all(-1)
    keep=finite&(opacity>.005)&(physical.max(-1).values<.3)
    if not keep.any():raise RuntimeError('Refinement would remove every primitive')
    average=gradient_sum/observation_count.clamp_min(1)
    candidates=torch.nonzero(keep&(average>2e-4),as_tuple=False).flatten()
    room=len(candidates) if max_points==0 else max(0,max_points-int(keep.sum()))
    priority_eligible=int(growth_priority_mask[candidates].sum()) if growth_priority_mask is not None else None
    if len(candidates)>room:
        ordered=candidates[torch.argsort(average[candidates],descending=True)]
        if growth_priority_mask is not None:
            selected=growth_priority_mask[ordered]
            ordered=torch.cat((ordered[selected],ordered[~selected]))
        candidates=ordered[:room]
    priority_added=int(growth_priority_mask[candidates].sum()) if growth_priority_mask is not None else None
    retained=torch.nonzero(keep,as_tuple=False).flatten();parents=torch.cat((retained,candidates))
    children=torch.arange(len(parents),device=parents.device)>=len(retained)
    offsets=torch.zeros(len(parents),3,device=parents.device);shrink=torch.zeros(len(parents),device=parents.device)
    split_parents_reset=0
    if len(candidates):
        split=physical[candidates].max(-1).values>surfel_split_scale;active=candidates[split]
        if len(active):
            scales3=physical[active]
            if scales3.shape[-1]==2:scales3=torch.cat((scales3,torch.zeros_like(scales3[:,:1])),1)
            noise=torch.randn_like(scales3)*scales3*.5
            displacement=(quaternion_rotation(parameters['quats'][active])@noise[...,None])[...,0]
            child_positions=len(retained)+torch.nonzero(split,as_tuple=False).flatten();retained_positions=torch.searchsorted(retained,active)
            offsets[retained_positions]=-displacement;offsets[child_positions]=displacement
            shrink[retained_positions]=math.log(1.6);shrink[child_positions]=math.log(1.6)
            if reset_split_parent_moments:
                # This slot now holds a moved, shrunken split descendant.
                # Its sibling already has children=True; both are new states.
                children[retained_positions]=True;split_parents_reset=len(active)
    replace_population(parameters,optimizer,family,parents,children,offsets,shrink)
    result=dict(before=n,after=len(parents),added=len(candidates),removed=n-len(retained))
    if reset_split_parent_moments:result['split_parents_reset']=split_parents_reset
    if growth_priority_mask is not None:result.update(priority_eligible=priority_eligible,priority_added=priority_added)
    return result


@torch.no_grad()
def reset_opacities(parameters,optimizer,family):
    parameter=parameters['opacities'];parameter.copy_(torch.logit(parameter.sigmoid().clamp_max(.01)))
    state=optimizer.state.get(parameter,{})
    for key in ['exp_avg','exp_avg_sq']:
        if key in state:state[key].zero_()


def auxiliary_refinement_transform(ray_transforms, screen_gradient):
    """Send only the auxiliary ray adjoint to gsplat's refinement dummy.

    The pinned primary backward uses dH[0:2,2] * H[2,2]. The dummy is an
    independent leaf, so a zero-valued connection leaves physical gradients
    unchanged while adding the missing auxiliary contribution to its gradient.
    """
    if ray_transforms.shape[-2:]!=(3,3) or screen_gradient.shape!=ray_transforms.shape[:-2]+(2,):
        raise ValueError('Auxiliary refinement needs matching unpacked ray/dummy shapes')
    if ray_transforms.device!=screen_gradient.device or ray_transforms.dtype!=screen_gradient.dtype:
        raise ValueError('Auxiliary refinement ray/dummy device and dtype must match')
    if not screen_gradient.is_leaf:
        raise ValueError('Refinement dummy must be independent of physical parameters')
    delta=(screen_gradient-screen_gradient.detach())*ray_transforms[...,2,2,None].detach()
    connection=torch.zeros_like(ray_transforms)
    connection[...,:2,2]=delta
    return ray_transforms+connection


def render_unbiased_surfels(parameters,camera_to_world,intrinsics,width,height,sh_degree,*,scene_radius=1.0,auxiliary=True,
                            world_to_camera=None,auxiliary_camera_to_world=None,loss_reduction='mean_segments',convergence_variant='adjacent',ray_depth_order=False,
                            auxiliary_refinement_gradient=False,surface_midpoint=False,front_convergence=False,convergence_physical_scale=1.,terminal_surface=False,primary_camera_to_world=None,coherent_midpoint=False,occlusion_aware_normal=False,physical_primary=False,dual_geometry=False,detached_alpha_normal=False):
    if type(detached_alpha_normal) is not bool:raise TypeError('detached_alpha_normal must be boolean')
    if type(dual_geometry) is not bool:raise TypeError('dual_geometry must be boolean')
    if dual_geometry and (physical_primary or coherent_midpoint or auxiliary_refinement_gradient or ray_depth_order or convergence_variant!='adjacent'):
        raise ValueError('Dual geometry requires original photo ordering and current-only adjacent geometry')
    if type(physical_primary) is not bool:raise TypeError('physical_primary must be boolean')
    if physical_primary and (coherent_midpoint or auxiliary_refinement_gradient or convergence_variant!='adjacent'):
        raise ValueError('Physical primary preserves the original adjacent/current-only/signed screen recipe')
    if type(occlusion_aware_normal) is not bool:raise TypeError('occlusion_aware_normal must be boolean')
    if type(coherent_midpoint) is not bool:raise TypeError('coherent_midpoint must be boolean')
    if coherent_midpoint and not surface_midpoint:raise ValueError('Coherent midpoint requires surface_midpoint')
    if type(terminal_surface) is not bool:raise TypeError('terminal_surface must be boolean')
    if type(front_convergence) is not bool:raise TypeError('front_convergence must be boolean')
    if not 0 < float(convergence_physical_scale) < float('inf'):raise ValueError('convergence_physical_scale must be finite and positive')
    if front_convergence and ((ray_depth_order and not physical_primary) or convergence_variant!='adjacent'):
        raise ValueError('Front-window convergence requires the regular adjacent training path')
    if type(surface_midpoint) is not bool:raise TypeError('surface_midpoint must be boolean')
    if type(ray_depth_order) is not bool:raise TypeError('ray_depth_order must be boolean')
    if ray_depth_order and auxiliary and not physical_primary:raise ValueError('This host ray-depth-order option is export-only; training stays unchanged')
    if convergence_variant not in ('adjacent','consensus'):raise ValueError('Unknown Unbiased convergence variant')
    if convergence_variant=='consensus' and loss_reduction!='mean_segments':raise ValueError('Experimental consensus is qualified only for mean_segments')
    from gsplat import rasterization_2dgs
    from ._cuda import cuda_available, rasterize_unbiased_aux
    from .depth_normal import depth_to_points
    view=torch.linalg.inv(camera_to_world) if world_to_camera is None else world_to_camera;tangent=parameters['scales'].exp()
    if tangent.shape[-1]!=2:raise ValueError('Unbiased trains exactly two tangent scales')
    scales=torch.cat((tangent,torch.ones_like(tangent[:,:1])),1)
    sh=torch.cat((parameters['sh0'],parameters['shN']),1)
    rasterizer=rasterization_2dgs
    physical_state=None;dual_state=None
    if dual_geometry:
        from .dual_geometry import dual_geometry_rasterizer
        rasterizer,dual_state=dual_geometry_rasterizer(rasterizer,midpoint=surface_midpoint,terminal=terminal_surface,front=front_convergence,quarter=1./convergence_physical_scale if front_convergence else scene_radius/4.)
    if physical_primary:
        from .physical_ordering import physical_primary_rasterizer
        rasterizer,physical_state=physical_primary_rasterizer(rasterizer,midpoint=surface_midpoint,
            terminal=terminal_surface,front=front_convergence,
            quarter=1./convergence_physical_scale if front_convergence else scene_radius/4.)
    if primary_camera_to_world is not None:
        from functools import partial
        from .cached_calibration import rasterization_2dgs_cached_camera
        if world_to_camera is None or auxiliary_camera_to_world is None:
            raise ValueError('Full primary calibration needs both original cached view inverses')
        rasterizer=partial(rasterization_2dgs_cached_camera,rasterizer,
            primary_inverse=primary_camera_to_world,linalg_inverse=auxiliary_camera_to_world)
    outputs=rasterizer(parameters['means'],parameters['quats'],scales,parameters['opacities'].sigmoid(),sh,
                              view[None],intrinsics[None],width,height,sh_degree=sh_degree,packed=False,
                              render_mode='RGB',near_plane=.01,distloss=False)
    rgb,alphas,normals=outputs[:3];meta=outputs[-1]
    ordered=None if physical_state is None else physical_state.pop('ordered')
    geometry_ordered=None if dual_state is None else dual_state.pop('geometry')
    photo_alphas,photo_normals=alphas,normals
    if geometry_ordered is not None:
        ordered=geometry_ordered
        alphas=ordered.alpha.reshape(1,height,width,1)
        normals=(ordered.normals @ view[:3,:3]).reshape(1,height,width,3)
    config=UnbiasedSurfelConfig(convergence_reduction=loss_reduction,normal_reduction=loss_reduction,surface_midpoint=surface_midpoint,front_convergence=front_convergence,terminal_surface=terminal_surface,coherent_midpoint=coherent_midpoint,detached_alpha_normal=dual_geometry or detached_alpha_normal)
    geometry_terms=None;consensus_stats=None
    if auxiliary:
        auxiliary_meta=meta
        if auxiliary_refinement_gradient:
            if convergence_variant!='adjacent':raise ValueError('Auxiliary refinement is qualified for the original adjacent loss only')
            auxiliary_meta=dict(meta,ray_transforms=auxiliary_refinement_transform(meta['ray_transforms'],meta['gradient_2dgs']))
        auxiliary_keywords={} if ordered is None else dict(ordered_outputs=ordered)
        aux=compute_from_gsplat_outputs(alphas,normals,auxiliary_meta,view[None],intrinsics[None],scene_radius=4./convergence_physical_scale if front_convergence else scene_radius,config=config,
            camera_to_world=auxiliary_camera_to_world,**auxiliary_keywords)
        depth=aux.corrected_depth.depth.reshape(height,width);valid=aux.corrected_depth.valid_mask.reshape(height,width)
        convergence,normal_loss=aux.convergence.loss,aux.normal_consistency.loss
        geometry_terms=dict(convergence=aux.convergence.per_segment,
            convergence_denominator=aux.convergence.denominator,
            normal=aux.normal_consistency.per_segment,normal_denominator=aux.normal_consistency.denominator)
        if occlusion_aware_normal:
            from .normal_support import continuous_normal_stencil
            normal_support=continuous_normal_stencil(depth,valid,max_depth_jump=.008).reshape(-1)
            geometry_terms['normal']=torch.where(normal_support,geometry_terms['normal'],0.)
            normal_loss=geometry_terms['normal'].sum()/geometry_terms['normal_denominator'].clamp_min(1.)
        if front_convergence:
            energy_scale=float(convergence_physical_scale)**2
            convergence=convergence*energy_scale
            geometry_terms['convergence']=geometry_terms['convergence']*energy_scale
        if convergence_variant=='consensus':
            from .ray_consensus import consensus_from_gsplat
            consensus_map,compact,wide=consensus_from_gsplat(meta,width,height,scene_radius=scene_radius)
            convergence=consensus_map.sum()/(width*height)
            geometry_terms['convergence']=consensus_map.reshape(-1)
            consensus_stats=dict(compact=compact,wide=wide)
    elif ordered is not None:
        depth=ordered.depth.reshape(height,width);valid=ordered.valid.reshape(height,width)
        convergence=depth.sum()*0;normal_loss=convergence
    else:
        if not cuda_available():raise RuntimeError('Compact evaluation requires the validated independent fused backend')
        depth,valid,_conv,_sel=rasterize_unbiased_aux(meta['means2d'].reshape(-1,2),meta['ray_transforms'].reshape(-1,9),
                    meta['opacities'].reshape(-1),meta['isect_offsets'].to(torch.int32),meta['flatten_ids'].to(torch.int32),
                    width,height,int(meta['tile_size']),.1,.6,scene_radius,1.25,
                    **(dict(ray_depth_order=True) if ray_depth_order else {}),
                    **(dict(surface_midpoint=True) if surface_midpoint else {}),
                    **(dict(terminal_surface=True) if terminal_surface else {}))
        depth,valid=depth[0],valid[0];convergence=depth.sum()*0;normal_loss=convergence
    points,point_valid=depth_to_points(depth[None],valid[None],camera_to_world[None],intrinsics[None])
    result=dict(rgb=rgb[0,...,:3],alpha=alphas[0,...,0],depth=depth,valid=valid&point_valid[0],
                points=points[0],normals=F.normalize(normals[0],dim=-1,eps=1e-12),meta=meta,
                convergence_loss=convergence,normal_loss=normal_loss,geometry_terms=geometry_terms)
    if geometry_ordered is not None:
        result['geometry']={key:result[key] for key in ('depth','valid','points','normals','alpha')}
        # Preserve the original auxiliary surface used by photo sorting.
        photo_depth,photo_valid,_conv,_sel=rasterize_unbiased_aux(meta['means2d'].reshape(-1,2),meta['ray_transforms'].reshape(-1,9),
            meta['opacities'].reshape(-1),meta['isect_offsets'].to(torch.int32),meta['flatten_ids'].to(torch.int32),
            width,height,int(meta['tile_size']),.1,.6,scene_radius,1.25,
            surface_midpoint=surface_midpoint,terminal_surface=terminal_surface)
        photo_points,photo_point_valid=depth_to_points(photo_depth,photo_valid,camera_to_world[None],intrinsics[None])
        result.update(alpha=photo_alphas[0,...,0],normals=F.normalize(photo_normals[0],dim=-1,eps=1e-12),
            depth=photo_depth[0],valid=photo_valid[0]&photo_point_valid[0],points=photo_points[0])
    if consensus_stats is not None:result['consensus_stats']=consensus_stats
    return result
