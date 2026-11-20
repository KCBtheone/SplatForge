"""Standalone PGSR, standalone Unbiased, then a commercial regional joint model.

SPDX-License-Identifier: Apache-2.0
The renderer base is only pinned Apache gsplat. All new math lives in modules.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback
import numpy as np
import torch
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from commercial_colmap import ColmapInputs,validate_normalization_resume,validate_resume_dataset_paths
from unbiased_gsplat.commercial_model import (initialize_parameters,make_optimizer,update_mean_learning_rate,
    refine_population,reset_opacities,render_unbiased_surfels,unbiased_regularizer_terms)
from unbiased_gsplat.planar_gaussian import render_planar_gaussians,edge_aware_normal_loss,PlanarGaussianConfig
from unbiased_gsplat.regional_commercial import compose_two_families,semantic_and_leakage,photometric_loss,physical_geometry_domain,select_opaque_geometry,native_radiance_anchor,regional_identity_policy
from unbiased_gsplat.dual_geometry import geometry_layers
from unbiased_gsplat.regional_commercial import native_visibility,normal_stencil_visibility,visible_native_geometry_losses
from unbiased_gsplat.multiview import PGSRMultiViewConfig,build_camera_neighbor_graph
from unbiased_gsplat.photometric import structural_similarity
from unbiased_gsplat.training_state import replay_sampler
from unbiased_gsplat.background import composite_background
from unbiased_gsplat.gradient_validation import (validate_adam_inputs,population_statistics_needed,geometry_regularizers_active,surfel_regularizer_activity,
    REFERENCE_SURFEL_NORMAL_START_STEP,REFERENCE_SURFEL_CONVERGENCE_START_STEP,
    validate_view_batch_size,batch_loss_for_backward,unscale_view_gradient,validate_view_batch_resume,validate_orientation_resume,validate_surface_midpoint_resume,validate_front_convergence_resume,validate_terminal_surface_resume,validate_coherent_midpoint_resume)
from unbiased_gsplat._cuda import get_build_info
from unbiased_gsplat.refinement_schedule import reset_opacities_if_due, opacity_reset_histories_match
from unbiased_gsplat.family_roles import initial_family_masks,ordered_native_layers,native_anchor_weights,selected_native_normals
from unbiased_gsplat.native_multiview import active_multiview_families,native_multiview_constraints
from unbiased_gsplat.view_tensor_cache import ViewTensorCache
from unbiased_gsplat import optimizer_execution
from unbiased_gsplat.optimizer_execution import adam_step
from unbiased_gsplat.sfm_surface_anchor import build_training_targets,surface_anchor_loss,surface_anchor_fields
from unbiased_gsplat.regional_identity_consistency import visible_identity_conflicts,RegionalIdentityEvidence
from unbiased_gsplat.input_photometric_consensus import input_photometric_consensus
from unbiased_gsplat.gradient_validation import validate_input_photo_consensus_resume
from unbiased_gsplat.physical_refinement import (NativePhysicalRefinement, phase4_view_statistic,
    validate_refinement_resume)


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,indent=2),encoding='utf-8');temporary.replace(path)


def gpu_processes():
    import subprocess
    lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True).strip()
    return [{'pid':int(s.split(',')[0]),'memory_mib':s.split(',')[1].strip()} for s in lines.splitlines() if s.strip()]


def production_disk_check(enabled):
    if enabled and Path('/root/autodl-tmp').is_dir() and shutil.disk_usage('/root/autodl-tmp').free < 20*2**30:
        raise RuntimeError('AutoDL data disk below20GiB; retain the existing checkpoint for recovery')


def tensors(item, *, background=None, camera_geometry_cache=False,full_camera_geometry_cache=False):
    result={k:torch.as_tensor(item[k],device='cuda') for k in ['foreground','hair','body','detail','unknown','known','K','c2w']}
    result['rgb']=torch.as_tensor(item['rgb'],device='cuda',dtype=torch.float32)/255
    if background is not None:
        result['rgb']=torch.where(result['foreground'][...,None],result['rgb'],background)
    if camera_geometry_cache:
        result['world_to_camera']=torch.linalg.inv(result['c2w'])
        result['intrinsics_inverse']=torch.linalg.inv(result['K'])
        # Keep the native adapter original inverse(inverse(c2w)) arithmetic.
        result['auxiliary_camera_to_world']=torch.linalg.inv(result['world_to_camera'][None])
    if full_camera_geometry_cache:
        from unbiased_gsplat.cached_calibration import prepare_primary_inverse
        if not camera_geometry_cache:raise ValueError('Full camera cache requires the original outer calibration cache')
        result['primary_camera_to_world']=prepare_primary_inverse(result['world_to_camera'])
    return result


def render(families,item,degree,auxiliary=True,appearance_weights=True,background=None,*,leakage_reduction='violators',hair_family='pgsr',occlusion_aware_identity=False,unbiased_reduction='mean_segments',joint_physical_visibility=False,defer_native_anchors=False,unbiased_convergence_variant='adjacent',export_ray_depth_order=False,unbiased_auxiliary_refinement_gradient=False,unbiased_surface_midpoint=False,unbiased_terminal_surface=False,unbiased_front_convergence=False,unbiased_convergence_physical_scale=None,regional_image_operator=None,regional_exposure=None,family_streams=None,unbiased_coherent_midpoint=False,surfel_selected_geometry=False,unbiased_occlusion_aware_normal=False,unbiased_physical_primary=False,unbiased_dual_geometry=False,unbiased_reference_regularizers=False):
    if type(unbiased_coherent_midpoint) is not bool:raise TypeError('unbiased_coherent_midpoint must be boolean')
    if unbiased_coherent_midpoint and not unbiased_surface_midpoint:raise ValueError('Coherent midpoint requires surface midpoint')
    if type(unbiased_front_convergence) is not bool:raise TypeError('unbiased_front_convergence must be boolean')
    if type(unbiased_terminal_surface) is not bool:raise TypeError('unbiased_terminal_surface must be boolean')
    if type(unbiased_surface_midpoint) is not bool:raise TypeError('unbiased_surface_midpoint must be boolean')
    if type(export_ray_depth_order) is not bool:raise TypeError('export_ray_depth_order must be boolean')
    if export_ray_depth_order and auxiliary:raise ValueError('Physical ray-depth ordering is export-only in this host entry')
    h,w=item['rgb'].shape[:2];layers={}
    if 'pgsr' in families:
        pgsr_renderer=family_streams.pgsr if family_streams is not None else render_planar_gaussians
        layers['pgsr']=pgsr_renderer(families['pgsr'],item['c2w'],item['K'],w,h,degree,
            world_to_camera=item.get('world_to_camera'),intrinsics_inverse=item.get('intrinsics_inverse'))
    if 'unbiased' in families:
        surfel_renderer=family_streams.surfel if family_streams is not None else render_unbiased_surfels
        variant_keywords={} if unbiased_convergence_variant=='adjacent' else dict(convergence_variant=unbiased_convergence_variant)
        if export_ray_depth_order:variant_keywords['ray_depth_order']=True
        if unbiased_auxiliary_refinement_gradient:variant_keywords['auxiliary_refinement_gradient']=True
        if unbiased_surface_midpoint:variant_keywords['surface_midpoint']=True
        if unbiased_terminal_surface:variant_keywords['terminal_surface']=True
        if unbiased_coherent_midpoint:variant_keywords['coherent_midpoint']=True
        if unbiased_occlusion_aware_normal:variant_keywords['occlusion_aware_normal']=True
        if unbiased_physical_primary:variant_keywords['physical_primary']=True
        if unbiased_dual_geometry:variant_keywords['dual_geometry']=True
        if unbiased_reference_regularizers and auxiliary:variant_keywords['detached_alpha_normal']=True
        if unbiased_front_convergence and auxiliary:
            if unbiased_convergence_physical_scale is None:raise ValueError('Front convergence requires the input normalization radius')
            variant_keywords.update(front_convergence=True,convergence_physical_scale=unbiased_convergence_physical_scale)
        layers['unbiased']=surfel_renderer(families['unbiased'],item['c2w'],item['K'],w,h,degree,auxiliary=auxiliary,
            world_to_camera=item.get('world_to_camera'),auxiliary_camera_to_world=item.get('auxiliary_camera_to_world'),primary_camera_to_world=item.get('primary_camera_to_world'),loss_reduction=unbiased_reduction,**variant_keywords)
    if surfel_selected_geometry and (len(layers)!=2 or not joint_physical_visibility or hair_family!='pgsr' or regional_image_operator is not None):
        raise ValueError('Selected Surfel geometry requires unfused joint PGSR-hair visibility')
    if regional_image_operator is not None:
        if len(layers)!=2 or not joint_physical_visibility or hair_family!='pgsr' or not occlusion_aware_identity or leakage_reduction!='trusted-support' or not defer_native_anchors:
            raise ValueError('Regional image fusion requires the current joint physical PGSR-hair training policy')
        pixel_layers={f:{k:layer[k] for k in ('rgb','alpha','depth','valid')} for f,layer in layers.items()}
        pixel_layers['unbiased']['geometry_terms']=layers['unbiased']['geometry_terms']
        image_item={k:item[k] for k in ('foreground','hair','body','detail','unknown','known','rgb')}
        composite,visibility,convergence,normal=regional_image_operator(pixel_layers['pgsr'],pixel_layers['unbiased'],image_item,background,regional_exposure)
        for family,layer in layers.items():layer['geometry_visibility']=visibility[family]
        layers['unbiased']['convergence_loss']=convergence;layers['unbiased']['normal_loss']=normal
        return composite,layers
    if joint_physical_visibility and len(layers)==2:
        selected_surfel=None
        if surfel_selected_geometry:
            selected_hair=((item['hair']>=.99)|(item['detail']>=.9)) & ~item['unknown']
            selected_surfel=(item['body']>=.99) & ~item['unknown'] & ~selected_hair
        visibility=native_visibility(geometry_layers(layers),hair_family,selected_surfel=selected_surfel)
        photo_visibility=native_visibility(layers,hair_family,selected_surfel=selected_surfel)
        for family,layer in layers.items():
            layer['geometry_visibility']=visibility[family]
            layer['photo_visibility']=photo_visibility[family]
        if auxiliary:
            layer=layers['unbiased'];layer['convergence_loss'],layer['normal_loss']=visible_native_geometry_losses(layer['geometry_terms'],visibility['unbiased'])
    if len(layers)==1:
        layer=next(iter(layers.values()))
        return dict(layer,rgb=composite_background(layer['rgb'],layer['alpha'],background)),layers
    trusted_hair=((item['hair']>=.99)|(item['detail']>=.9)) & ~item['unknown']
    trusted_body=(item['body']>=.99) & ~item['unknown'] & ~trusted_hair
    weights=None
    if appearance_weights:
        policy=regional_identity_policy(item['foreground'],item['hair'],item['known'],item['unknown'],
                                       trusted_hair=trusted_hair,trusted_body=trusted_body)
        weights=policy['appearance_weights']
    semantic_hair,semantic_body=ordered_native_layers(layers,hair_family)
    composite=compose_two_families(semantic_hair,semantic_body,trusted_hair=trusted_hair,trusted_body=trusted_body,
                                  background=background,appearance_weights=weights,
                                  occlusion_aware_identity=occlusion_aware_identity)
    semantic,leakage=semantic_and_leakage(composite,trusted_hair,trusted_body,leakage_reduction=leakage_reduction)
    composite.update(semantic_loss=semantic,leakage_loss=leakage,trusted_hair=trusted_hair,trusted_body=trusted_body)
    if appearance_weights:
        anchors=native_anchor_weights(policy,hair_family)
        composite['native_anchor_weights']=anchors
        if not defer_native_anchors:
            pg_anchor=native_radiance_anchor(layers['pgsr'],item['rgb'],anchors['pgsr'],background=background)
            ug_anchor=native_radiance_anchor(layers['unbiased'],item['rgb'],anchors['unbiased'],background=background)
            composite.update(native_pgsr_anchor=pg_anchor,native_unbiased_anchor=ug_anchor)
    return composite,layers


def checkpoint(path,families,optimizer,exposure,updates,inputs,args,training_state=None):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    stage = staged_updates(args, updates)
    torch.save({'format':'commercial_regional_v1','mode':args.mode,'actual_optimizer_updates':updates,
                **stage,
                'families':{f:{k:v.detach().cpu() for k,v in p.items()} for f,p in families.items()},
                'optimizer_state':optimizer.state_dict(),'exposure':exposure.detach().cpu() if exposure is not None else None,
                'normalization_center':inputs.center,'normalization_radius':inputs.radius,
                'seed':args.seed,'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all(),
                'training_state':training_state,
                'config':vars(args),'source_snapshot':None if getattr(args,'production',False) else ((ROOT/'SNAPSHOT_SHA256').read_text().strip() if (ROOT/'SNAPSHOT_SHA256').exists() else None)},temporary)
    temporary.replace(path)


def staged_updates(args, updates):
    if getattr(args,'initialization_strategy',None) == 'regional_joint_from_sfm':
        return dict(initialization_strategy=args.initialization_strategy,
                    updates_by_family={'pgsr':updates,'unbiased':updates})
    if getattr(args,'initialization_strategy',None) != 'pgsr_surface_warmstart':
        return {}
    warmup=args.warmup_updates
    if not warmup <= updates <= args.max_steps:
        raise ValueError('Invalid staged optimizer update count')
    return dict(initialization_strategy=args.initialization_strategy,warmup_updates=warmup,
                joint_updates=updates-warmup,updates_by_family={'pgsr':updates,'unbiased':updates-warmup})


@torch.no_grad()
def export_and_evaluate(families,inputs,degree,out,deadline,mode,background=None,*,voxel_length=.002,sdf_trunc=.008,hair_family='pgsr',production=False,export_ray_depth_order=False,unbiased_surface_midpoint=False,unbiased_terminal_surface=False,unbiased_physical_primary=False,unbiased_dual_geometry=False,studio=False):
    if type(unbiased_terminal_surface) is not bool:raise TypeError('unbiased_terminal_surface must be boolean')
    if type(unbiased_surface_midpoint) is not bool:raise TypeError('unbiased_surface_midpoint must be boolean')
    if type(export_ray_depth_order) is not bool:raise TypeError('export_ray_depth_order must be boolean')
    import open3d as o3d
    output=out/'geometry';output.mkdir(exist_ok=production)
    volume=o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=voxel_length,sdf_trunc=sdf_trunc,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    from unbiased_gsplat.dual_geometry import geometry_layers
    review_stems={'00060','00100','00112','00120','00121','00122','00131','00144','00149'}
    records=[];evaluations=[];integrated=0
    for index in range(len(inputs.rows)):
        production_disk_check(production and not studio)
        if time.time()>deadline-12:break
        row=inputs.rows[index];item=tensors(inputs.load(index),background=background if production else None);result,layers=render(families,item,degree,auxiliary=False,appearance_weights=False,background=background,hair_family=hair_family,
            **(dict(export_ray_depth_order=True) if export_ray_depth_order else {}),
            **(dict(unbiased_surface_midpoint=True) if unbiased_surface_midpoint else {}),
            **(dict(unbiased_terminal_surface=True) if unbiased_terminal_surface else {}),
            **(dict(unbiased_physical_primary=True) if unbiased_physical_primary else {}),
            **(dict(unbiased_dual_geometry=True) if unbiased_dual_geometry else {}))
        export_layers=geometry_layers(layers)
        geometry=select_opaque_geometry(*ordered_native_layers(export_layers,hair_family)) if mode=='joint' else next(iter(export_layers.values()))
        alpha=geometry['alpha'];valid=geometry['valid'] & item['foreground'] & (alpha>=.5)
        if mode=='joint':valid=valid & ~item['unknown']
        depth=torch.where(valid,geometry['depth'],torch.zeros_like(geometry['depth'])).cpu().numpy().astype(np.float32)
        rgb=(item['rgb'].clamp(0,1)*255).byte().cpu().numpy()
        stem=Path(row['name']).stem
        if not production:
            np.savez_compressed(output/(stem+'.npz'),depth=depth,valid=valid.cpu().numpy(),alpha=alpha.cpu().numpy(),
                                K=item['K'].cpu().numpy(),c2w=item['c2w'].cpu().numpy())
        if stem in review_stems and not production:
            predicted=(result['rgb'].clamp(0,1)*255).byte().cpu().numpy()
            normal=next(iter(export_layers.values()))['normals'] if len(layers)==1 else selected_native_normals(export_layers,geometry,hair_family)
            normal_image=((normal.clamp(-1,1)+1)*127.5).byte().cpu().numpy();normal_image[~valid.cpu().numpy()]=0
            board=np.concatenate((rgb,predicted,normal_image),axis=1)
            Image.fromarray(board).save(output/(stem+'_review.png'))
        if index in inputs.test_indices and not production:
            foreground=item['foreground'];error=(result['rgb']-item['rgb']).square().mean(-1)
            mse=(error*foreground).sum()/foreground.sum().clamp_min(1)
            evaluations.append(dict(frame=row['name'],masked_psnr=float(-10*torch.log10(mse.clamp_min(1e-12))),
                                    full_ssim=float(structural_similarity(result['rgb'].clamp(0,1),item['rgb'])),
                                    foreground_depth_coverage=float(valid.sum()/foreground.sum().clamp_min(1))))
        if valid.any():
            h,w=depth.shape;k=item['K'].cpu().numpy()
            intrinsic=o3d.camera.PinholeCameraIntrinsic(w,h,float(k[0,0]),float(k[1,1]),float(k[0,2]),float(k[1,2]))
            rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(rgb.copy()),o3d.geometry.Image(depth.copy()),depth_scale=1.,depth_trunc=20.,convert_rgb_to_intensity=False)
            volume.integrate(rgbd,intrinsic,np.linalg.inv(item['c2w'].cpu().numpy()));integrated+=1
        records.append(dict(frame=row['name'],valid_pixels=int(valid.sum()),alpha_threshold=.5))
        if index%20==0:print(json.dumps({'phase':'export','mode':mode,'view':index,'total':len(inputs.rows),'integrated':integrated}),flush=True)
        del result,layers,item
    mesh=volume.extract_triangle_mesh();mesh.compute_vertex_normals()
    # Preserve raw TSDF and original-frame output; do not crop, cap or repair.
    path = out / "mesh.ply" if studio else output / "raw_tsdf_normalized.ply"
    if studio:
        if not len(mesh.triangles):
            raise RuntimeError("Unbiased Surfel TSDF extraction produced no surface")
        temporary = path.with_suffix(".tmp.ply")
        if not o3d.io.write_triangle_mesh(str(temporary), mesh):
            raise RuntimeError(f"Could not write Unbiased Surfel mesh to {path}")
        temporary.replace(path)
    else:
        o3d.io.write_triangle_mesh(str(path), mesh)
    if not production:
        vertices=np.asarray(mesh.vertices);vertices*=inputs.radius;vertices+=inputs.center
        o3d.io.write_triangle_mesh(str(output/'raw_tsdf_colmap.ply'),mesh)
    report=dict(views_exported=len(records),expected_views=len(inputs.rows),integrated_views=integrated,
                full_export=len(records)==len(inputs.rows),vertices=len(mesh.vertices),faces=len(mesh.triangles),
                evaluation=evaluations,views=records,geometry_accepted=False,
                voxel_length=voxel_length,sdf_trunc=sdf_trunc,coordinate_frame='normalized COLMAP world',physical_metric_scale_known=False,
                export_ray_depth_order=export_ray_depth_order,unbiased_surface_midpoint=unbiased_surface_midpoint,unbiased_terminal_surface=unbiased_terminal_surface)
    write_json(output/'geometry_report.json',report)
    return report


def probe(out,background=None):
    """Real pinned CUDA forward/backward for both distinct representations."""
    points=np.array([[x,y,2+.08*x] for x in np.linspace(-.5,.5,9) for y in np.linspace(-.5,.5,9)],dtype=np.float32)
    colors=np.random.default_rng(42).uniform(.2,.8,points.shape).astype(np.float32)
    families={f:initialize_parameters(points,colors,axes=axes) for f,axes in [('pgsr',3),('unbiased',2)]}
    for params in families.values():
        with torch.no_grad():params['opacities'].fill_(2.)
    item=dict(rgb=torch.full((40,40,3),.4,device='cuda'),c2w=torch.eye(4,device='cuda'),K=torch.tensor([[45.,0,20],[0,45,20],[0,0,1]],device='cuda'),
              foreground=torch.ones(40,40,dtype=torch.bool,device='cuda'),hair=torch.zeros(40,40,device='cuda'),body=torch.ones(40,40,device='cuda'),
              detail=torch.zeros(40,40,device='cuda'),unknown=torch.zeros(40,40,dtype=torch.bool,device='cuda'),known=torch.ones(40,40,dtype=torch.bool,device='cuda'))
    rows=[]
    for family in families:
        prediction,layers=render({family:families[family]},item,1,background=background)
        loss=photometric_loss(prediction['rgb'],item['rgb'])
        if family=='pgsr':loss=loss+.1*prediction['depth'].mean()+prediction['flatten_loss']
        else:loss=loss+7*prediction['convergence_loss']+.05*prediction['normal_loss']+.1*prediction['depth'].mean()
        loss.backward()
        gradients={key:{'finite':bool(p.grad is not None and torch.isfinite(p.grad).all()),'norm':float(p.grad.norm()) if p.grad is not None else 0.} for key,p in families[family].items()}
        if not all(v['finite'] and v['norm']>0 for v in gradients.values()):raise RuntimeError('Independent native CUDA gradient probe failed: '+family+str(gradients))
        rows.append(dict(family=family,axes=families[family]['scales'].shape[1],gradients=gradients,valid_pixels=int(prediction['valid'].sum())))
        for p in families[family].values():p.grad=None
    result=dict(status='PASS_PINNED_GPU_COMPONENTS_NOT_GEOMETRY',rows=rows,gsplat_base_only=True,geometry_accepted=False)
    write_json(out/'GPU_COMPONENT_PROBE.json',result);print(json.dumps(result),flush=True)


def export_studio_checkpoint(args):
    saved = torch.load(args.export_only, map_location="cpu", weights_only=False)
    if saved.get("format") != "commercial_regional_v1" or saved.get("mode") != "unbiased":
        raise ValueError("Expected an Unbiased Surfel checkpoint")
    recipe = saved["config"]
    validate_resume_dataset_paths(recipe, args.data_dir, recipe.get("labels"), args.output)
    inputs = ColmapInputs(
        args.data_dir, labels=recipe.get("labels"),
        label_subject=recipe.get("label_subject", "subject3"),
        factor=recipe["factor"], cache_views=recipe.get("cache_views", 12),
    )
    validate_normalization_resume(saved["normalization_center"], saved["normalization_radius"],
                                  inputs.center, inputs.radius)
    families = {"unbiased": {name: value.cuda() for name, value in saved["families"]["unbiased"].items()}}
    background = torch.full((3,), float(recipe.get("background") == "white"), device="cuda")
    degree = min(3, max(0, (saved["actual_optimizer_updates"] - 1) // 1000))
    geometry = export_and_evaluate(
        families, inputs, degree, Path(args.output), float("inf"), "unbiased", background,
        voxel_length=args.tsdf_voxel, sdf_trunc=args.tsdf_trunc, production=True, studio=True,
        export_ray_depth_order=recipe.get("export_ray_depth_order", False),
        unbiased_surface_midpoint=recipe.get("unbiased_surface_midpoint", False),
        unbiased_terminal_surface=recipe.get("unbiased_terminal_surface", False),
        unbiased_physical_primary=recipe.get("unbiased_physical_primary", False),
        unbiased_dual_geometry=recipe.get("unbiased_dual_geometry", False),
    )
    print(json.dumps({key: geometry[key] for key in ("vertices", "faces", "integrated_views")}), flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['pgsr','unbiased','joint','probe'],required=True)
    parser.add_argument('--data-dir');parser.add_argument('--labels');parser.add_argument('--output',required=True)
    parser.add_argument('--label-subject',default='subject3')
    parser.add_argument('--tsdf-voxel',type=float,default=.002);parser.add_argument('--tsdf-trunc',type=float,default=.008)
    parser.add_argument('--seconds',type=float,default=0);parser.add_argument('--max-steps',type=int,default=30000)
    parser.add_argument('--seed',type=int,default=42);parser.add_argument('--max-points',type=int,default=0,
                        help='Optional primitive count cap per family; 0 disables count-based subsampling and growth truncation')
    parser.add_argument('--factor',type=int,default=1);parser.add_argument('--memory-fraction',type=float,default=.24)
    parser.add_argument('--view-batch-size',type=int,choices=[1,2],default=1,
                        help='Experimental original-view loss averaging per Adam update; default1 preserves existing recipe')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--resume',type=Path);parser.add_argument('--save-interval',type=int,default=3000)
    parser.add_argument('--background',choices=['black','white'],default='black')
    parser.add_argument('--cache-views',type=int,default=12)
    parser.add_argument('--opacity-reset-policy',choices=['legacy','refinement'],default='legacy')
    parser.add_argument('--leakage-reduction',choices=['violators','trusted-support'],default='violators')
    parser.add_argument('--pgsr-normal-weight',type=float,default=.015)
    parser.add_argument('--hair-family',choices=['pgsr','unbiased'],default='pgsr')
    parser.add_argument('--multiview-families',choices=['pgsr','both'],default='pgsr')
    parser.add_argument('--regional-identity-consistency',action='store_true')
    parser.add_argument('--unbiased-normal-weight',type=float,default=.05)
    parser.add_argument('--unbiased-convergence-weight',type=float,default=7.)
    parser.add_argument('--unbiased-convergence-variant',choices=['adjacent','consensus'],default='adjacent',
                        help='Experimental compact-ray consensus; original adjacent convergence remains default')
    parser.add_argument('--unbiased-start-iter',type=int,default=1000,
                        help='Experimental Unbiased geometry activation timing; default preserves original1000, PGSR unchanged')
    parser.add_argument('--unbiased-orientation-init',choices=['uniform_sphere','positive_uniform','sfm_plane'],default='uniform_sphere',
                        help='Optional independent raw Surfel orientation prior; legacy default retained pending quality validation')
    parser.add_argument('--unbiased-sfm-anchor-weight',type=float,default=0.,
                        help='Experimental additional input-SfM surface constraint; default off, not an author equation')
    parser.add_argument('--unbiased-sfm-anchor-domain',choices=['native_surfel','opaque_scene'],default='native_surfel',
                        help='Optional joint opaque-scene input constraint; native Surfel remains historical default')
    parser.add_argument('--unbiased-reduction',choices=['mean_segments','mean_active_segments'],default='mean_segments')
    parser.add_argument('--unbiased-auxiliary-refinement-gradient',action='store_true',
                        help='Include existing auxiliary depth/convergence adjoints in the Surfel refinement dummy')
    parser.add_argument('--unbiased-refinement-statistic',choices=['signed','phase4'],default='signed',
                        help='Experimental physical pixel-phase refinement demand; signed remains default')
    parser.add_argument('--unbiased-surface-midpoint',action='store_true',
                        help='Use the qualified adjacent surface midpoint with unchanged current-hit depth VJP')
    parser.add_argument('--unbiased-terminal-surface',action='store_true',
                        help='Retain the qualified terminal surface when no corrected-opacity crossing occurs')
    parser.add_argument('--unbiased-input-photo-consensus',action='store_true',
                        help='Experimental input-only median photo across existing camera neighbors')
    parser.add_argument('--unbiased-front-convergence',action='store_true',
                        help='Use front-window convergence in the normalized camera-radius coordinate system')
    parser.add_argument('--unbiased-dual-geometry',action='store_true',help='Unqualified independent candidate-order geometry, retaining original photo fields')
    parser.add_argument('--unbiased-reference-regularizers',action='store_true',
        help='Use detached-alpha Surfel normal loss and separate normal/convergence start steps')
    parser.add_argument('--unbiased-physical-primary',action='store_true',help='Experimental shared per-ray physical primary and auxiliary ordering; raw-SfM training only')
    parser.add_argument('--unbiased-occlusion-aware-normal',action='store_true',help='Experimental native-depth continuity support for the original normal stencil')
    parser.add_argument('--unbiased-coherent-midpoint',action='store_true',
                        help='Differentiate the unchanged adjacent midpoint through both contributing hits')
    parser.add_argument('--native-anchor-color-space',choices=['raw','exposure'],default='exposure')
    parser.add_argument('--single-surfel-exposure',action='store_true',help='Diagnostic single-Surfel parity with the existing PGSR scalar exposure model')
    parser.add_argument('--camera-geometry-cache',action='store_true')
    parser.add_argument('--full-camera-geometry-cache',action='store_true')
    parser.add_argument('--view-tensor-cache',choices=['none','gpu'],default='none')
    parser.add_argument('--view-cache-palette',action='store_true',help='Store immutable original input values in lossless per-field palettes')
    parser.add_argument('--parallel-family-streams',action='store_true',help='Opt-in qualified native PGSR/Surfel fork-join training schedule')
    parser.add_argument('--parallel-surfel-auxiliary',action='store_true')
    parser.add_argument('--production',action='store_true',help='Use the clean worker contract: explicit native CUDA and minimal exports, no source/result hashes')
    parser.add_argument('--export-ray-depth-order',action='store_true',help='Use existing physical ray-depth ordering for export only; training is unchanged')
    parser.add_argument('--regional-image-fusion',action='store_true')
    parser.add_argument('--regional-image-cudagraph',action='store_true')
    parser.add_argument('--tensor-loss-fusion',action='store_true')
    parser.add_argument('--loss-cudagraphs',action='store_true')
    parser.add_argument('--separable-ssim',action='store_true')
    parser.add_argument('--occlusion-aware-identity',action='store_true')
    parser.add_argument('--joint-physical-visibility',action='store_true')
    parser.add_argument('--surfel-selected-geometry',action='store_true',help='Keep valid photo-selected body Surfel geometric supervision')
    parser.add_argument('--aux-depth-vjp-carry',action='store_true')
    parser.add_argument('--aux-front-prefix-replay',action='store_true')
    parser.add_argument('--aux-front-prefix-forward',action='store_true')
    parser.add_argument('--joint-group-adam',action='store_true',help='Opt-in same Adam update across all original parameter groups')
    parser.add_argument('--native-point-normals',action='store_true')
    parser.add_argument('--shared-surfel-forward',action='store_true')
    parser.add_argument('--surfel-last-backward',action='store_true')
    parser.add_argument('--zero-adjoint-tiles',action='store_true')
    parser.add_argument("--studio", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cuda-source", type=Path, default=ROOT / "csrc")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--no-export", action="store_true")
    action.add_argument("--export-only", type=Path, metavar="CHECKPOINT")
    args=parser.parse_args()
    if (args.studio or args.export_only) and not (args.studio and args.production and args.mode == "unbiased"):
        parser.error("Studio requires --production --mode unbiased")
    if args.studio:
        from unbiased_gsplat import _cuda

        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Unbiased Surfel requires a CUDA device")
        if device.index is not None:
            torch.cuda.set_device(device)
        _cuda.load_studio_cuda(args.cuda_source)
        if not args.no_export:
            import open3d
        if args.export_only:
            export_studio_checkpoint(args)
            return
    args.unbiased_convergence_units='normalized_camera_radius'
    reference_regularizers=args.unbiased_dual_geometry or args.unbiased_reference_regularizers
    if args.unbiased_reference_regularizers and (args.mode=='pgsr' or args.unbiased_start_iter!=1000):
        parser.error('Reference Surfel regularizers require Surfel and their fixed start schedule')
    if args.unbiased_dual_geometry and args.export_ray_depth_order:
        parser.error('Dual geometry already supplies export depth in candidate order; omit --export-ray-depth-order')
    if args.max_points<0:
        parser.error('--max-points must be nonnegative; 0 means no primitive count cap')
    if args.zero_adjoint_tiles and (args.mode!='joint' or args.surfel_last_backward or args.parallel_family_streams or args.parallel_surfel_auxiliary):
        raise ValueError('Zero-adjoint tiles require the original serial joint backward kernels')
    if args.production and args.max_points!=0:
        parser.error('Production --max-points must be 0; primitive count caps are not authorized')
    if args.full_camera_geometry_cache and (not args.camera_geometry_cache or not args.shared_surfel_forward or args.mode!='joint' or args.view_tensor_cache!='gpu'):
        raise ValueError('Full camera cache requires the qualified joint shared-forward and GPU immutable-view cache')
    from unbiased_gsplat import cached_calibration
    cached_calibration.reset_calls()
    if args.surfel_last_backward and not args.shared_surfel_forward:
        raise ValueError('Last-contributor backward requires the qualified shared Surfel forward scope')
    if args.shared_surfel_forward and (args.mode!='joint' or args.parallel_family_streams or args.parallel_surfel_auxiliary or args.aux_front_prefix_forward):
        raise ValueError('Shared Surfel forward requires the original serial joint execution')
    if args.parallel_surfel_auxiliary and (args.mode!='joint' or args.parallel_family_streams or args.aux_front_prefix_forward):
        parser.error('--parallel-surfel-auxiliary requires the current joint ordinary-forward profile without family streams')
    if args.parallel_family_streams and args.mode!='joint':
        parser.error('--parallel-family-streams requires both joint families')
    if args.view_cache_palette and args.view_tensor_cache!='gpu':
        parser.error('--view-cache-palette requires --view-tensor-cache gpu')
    validate_view_batch_size(args.view_batch_size)
    if args.unbiased_refinement_statistic=='phase4' and (args.mode!='unbiased' or args.view_batch_size!=1
            or args.shared_surfel_forward or args.parallel_family_streams or args.parallel_surfel_auxiliary
            or args.zero_adjoint_tiles or args.unbiased_auxiliary_refinement_gradient):
        raise ValueError('Phase4 currently requires ordinary single-Surfel batch1 primary refinement')
    if args.unbiased_coherent_midpoint and (not args.unbiased_surface_midpoint or args.mode=='pgsr' or args.aux_front_prefix_replay):
        raise ValueError('Coherent midpoint requires Surfel midpoint with generic auxiliary backward')
    if args.unbiased_dual_geometry and (args.view_batch_size!=1 or args.unbiased_refinement_statistic!='signed' or args.aux_depth_vjp_carry or args.aux_front_prefix_replay or args.aux_front_prefix_forward or args.zero_adjoint_tiles or args.mode=='pgsr' or args.unbiased_physical_primary or args.shared_surfel_forward or args.parallel_family_streams or args.parallel_surfel_auxiliary or args.surfel_last_backward or args.unbiased_coherent_midpoint or args.unbiased_auxiliary_refinement_gradient or args.unbiased_convergence_variant!='adjacent' or args.regional_image_fusion or args.regional_image_cudagraph):
        raise ValueError('Dual geometry requires ordinary serial rendering and unfused adjacent geometry')
    if args.unbiased_physical_primary and (args.mode=='pgsr' or args.unbiased_refinement_statistic!='signed' or args.unbiased_coherent_midpoint or args.unbiased_auxiliary_refinement_gradient):
        raise ValueError('Physical primary requires Surfel with original signed refinement and adjacent current-only auxiliary recipe')
    if args.unbiased_physical_primary and (args.shared_surfel_forward or args.surfel_last_backward
            or args.parallel_family_streams or args.parallel_surfel_auxiliary or args.aux_depth_vjp_carry
            or args.aux_front_prefix_replay or args.aux_front_prefix_forward or args.zero_adjoint_tiles):
        raise ValueError('Physical primary requires the ordinary serial renderer without legacy shared/auxiliary execution switches')
    if args.unbiased_occlusion_aware_normal and (args.mode=='pgsr' or not args.unbiased_surface_midpoint):
        raise ValueError('Occlusion-aware normal support requires corrected Surfel surface midpoint')
    if args.mode=='pgsr' and args.unbiased_orientation_init!='uniform_sphere':
        raise ValueError('Alternative orientation initialization is Surfel-only')
    if args.mode=='pgsr' and args.unbiased_terminal_surface:
        raise ValueError('Terminal Surfel surface is unavailable in PGSR-only mode')
    if args.mode=='pgsr' and args.unbiased_surface_midpoint:
        raise ValueError('Surface midpoint is Surfel-only')
    if args.unbiased_front_convergence and (args.mode=='pgsr' or args.unbiased_convergence_variant!='adjacent'):
        raise ValueError('Front convergence requires a Surfel family with the adjacent loss')
    if args.regional_image_fusion and args.regional_image_cudagraph:
        raise ValueError('Choose only one regional image execution mode')
    regional_image_enabled=args.regional_image_fusion or args.regional_image_cudagraph
    if args.surfel_selected_geometry and (args.mode!='joint' or args.hair_family!='pgsr' or not args.joint_physical_visibility or regional_image_enabled):
        raise ValueError('Selected Surfel geometry requires unfused joint PGSR-hair visibility')
    if args.regional_identity_consistency and (args.mode!='joint' or not args.joint_physical_visibility):
        raise ValueError('Cross-view identity abstention requires simultaneous joint physical geometry')
    if args.unbiased_input_photo_consensus and (args.mode!='joint' or args.hair_family!='pgsr' or args.multiview_families!='pgsr'):
        raise ValueError('Input photo consensus requires joint PGSR-hair with original PGSR-only native multiview')
    if regional_image_enabled and (args.mode!='joint' or args.hair_family!='pgsr' or not args.joint_physical_visibility or not args.occlusion_aware_identity or args.leakage_reduction!='trusted-support' or args.native_anchor_color_space!='exposure' or args.tensor_loss_fusion or args.separable_ssim or args.loss_cudagraphs):
        raise ValueError('Regional image fusion requires current joint PGSR-hair physical/exposure policy and the original SSIM')
    consensus_before=None
    if args.unbiased_convergence_variant=='consensus':
        if args.mode not in ('unbiased','joint') or args.unbiased_reduction!='mean_segments':raise ValueError('Experimental consensus requires an Unbiased family and mean_segments')
        from unbiased_gsplat.ray_consensus import NATIVE_CALLS
        consensus_before=dict(NATIVE_CALLS)
    if args.single_surfel_exposure and args.mode!='unbiased':
        raise ValueError('--single-surfel-exposure is only valid for the standalone unbiased diagnostic')
    from unbiased_gsplat import _cuda, depth_normal
    if args.aux_front_prefix_replay and (args.mode!='joint' or args.aux_depth_vjp_carry or
            not args.unbiased_front_convergence or not args.unbiased_surface_midpoint or not args.unbiased_terminal_surface):
        raise ValueError('Front-prefix replay requires the current joint midpoint/front/terminal profile without depth carry')
    _cuda.AUX_FRONT_PREFIX_REPLAY=args.aux_front_prefix_replay
    _cuda.AUX_FRONT_PREFIX_REPLAY_CALLS=0
    if args.aux_front_prefix_forward and (args.mode!='joint' or not args.unbiased_front_convergence):
        raise ValueError('Exact front-prefix forward requires the current joint front-convergence profile')
    _cuda.AUX_FRONT_PREFIX_FORWARD=args.aux_front_prefix_forward
    _cuda.AUX_FRONT_PREFIX_FORWARD_CALLS=0
    depth_normal.NATIVE_POINT_NORMALS=args.native_point_normals
    depth_normal.NATIVE_POINT_NORMAL_CALLS=0
    if args.aux_depth_vjp_carry and not getattr(_cuda.unbiased_gsplat_cuda,'optional_aux_depth_vjp_carry',False):
        raise RuntimeError('Requested auxiliary depth VJP carry requires the current owned native extension')
    _cuda.AUX_DEPTH_VJP_CARRY=args.aux_depth_vjp_carry
    _cuda.AUX_DEPTH_VJP_CARRY_CALLS=0
    if args.loss_cudagraphs and not args.tensor_loss_fusion:raise ValueError('SSIM CUDA graphs require tensor loss fusion')
    if args.view_batch_size>1 and args.loss_cudagraphs:raise ValueError('Batched views with SSIM CUDA graphs are not qualified')
    if not np.isfinite(args.unbiased_normal_weight) or args.unbiased_normal_weight<0:raise ValueError('Invalid Unbiased normal weight')
    if not np.isfinite(args.unbiased_convergence_weight) or args.unbiased_convergence_weight<0:raise ValueError('Invalid Unbiased convergence weight')
    if args.unbiased_start_iter<0:raise ValueError('Unbiased start iteration must be nonnegative')
    if not np.isfinite(args.unbiased_sfm_anchor_weight) or args.unbiased_sfm_anchor_weight<0:raise ValueError('Invalid input-SfM anchor weight')
    if args.unbiased_sfm_anchor_domain=='opaque_scene' and args.mode!='joint':
        raise ValueError('Opaque-scene SfM constraints require simultaneous joint mode')
    if args.unbiased_sfm_anchor_weight>0 and args.mode not in ['unbiased','joint']:
        raise ValueError('Input-SfM surface anchors require the Unbiased family')
    if not all(np.isfinite(v) and v>0 for v in [args.tsdf_voxel,args.tsdf_trunc]):raise ValueError('TSDF voxel and truncation must be finite and positive')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=args.production)
    start=time.time();deadline=start+args.seconds if args.seconds>0 else float('inf')
    optimizer_execution.GROUP_KERNEL_CALLS=0;optimizer_execution.ORIGINAL_INITIALIZATION_CALLS=0
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    background=torch.full((3,),1. if args.background=='white' else 0.,device='cuda')
    if args.mode=='probe':probe(out,background);return
    from functools import partial
    from torch._inductor import metrics as fusion_metrics
    kernels_before=int(fusion_metrics.generated_kernel_count)
    ssim_operator=partial(structural_similarity,separable=True) if args.separable_ssim else structural_similarity
    normal_operator=edge_aware_normal_loss
    graph_replays=0;ssim_forward_replays=0;loss_calls=dict(ssim=0,normal=0)
    if args.loss_cudagraphs:
        from torch._dynamo.utils import counters as compiler_counters
        from torch._inductor.cudagraph_trees import CUDAGraphNode
        if not hasattr(torch.compiler,'cudagraph_mark_step_begin'):raise RuntimeError('SSIM graphs require productionTorch2.7 API')
        skips_before=int(compiler_counters['inductor'].get('cudagraph_skips',0))
        original_run_graph=CUDAGraphNode.run_graph
        def counted_run_graph(node,*values,**keywords):
            nonlocal graph_replays
            result=original_run_graph(node,*values,**keywords);graph_replays+=1;return result
        CUDAGraphNode.run_graph=counted_run_graph
    if args.tensor_loss_fusion:
        compiled_ssim=torch.compile(ssim_operator,backend='inductor',fullgraph=False,mode='reduce-overhead' if args.loss_cudagraphs else 'default')
        compiled_normal=torch.compile(normal_operator,backend='inductor',fullgraph=False,mode='default')
        def ssim_operator(*values,**keywords):
            nonlocal ssim_forward_replays
            loss_calls['ssim']+=1;before=graph_replays
            result=compiled_ssim(*values,**keywords);ssim_forward_replays+=graph_replays-before;return result
        def normal_operator(*values,**keywords):
            loss_calls['normal']+=1;return compiled_normal(*values,**keywords)
    photo_operator=partial(photometric_loss,structural_similarity_function=ssim_operator)
    regional_image_operator=None;regional_image_calls=0
    if args.regional_image_cudagraph:
        from unbiased_gsplat.regional_image_graph import OriginalImageObjectiveGraph
        regional_image_operator=OriginalImageObjectiveGraph()
    elif args.regional_image_fusion:
        from unbiased_gsplat.regional_image_fusion import joint_image_objective
        def compile_required_image(*values):
            if not torch.compiler.is_compiling():raise RuntimeError('Regional image fusion cannot fall back to eager')
            return joint_image_objective(*values)
        compiled_regional_image=torch.compile(compile_required_image,backend='inductor',fullgraph=True,dynamic=False,mode='default')
        def regional_image_operator(*values):
            nonlocal regional_image_calls
            regional_image_calls+=1;return compiled_regional_image(*values)
    native_calls=[0]
    physical_primary_runtime=None
    dual_geometry_runtime=None
    if args.unbiased_dual_geometry:
        from unbiased_gsplat import dual_geometry as dual_geometry_runtime
        dual_geometry_calls_before=dual_geometry_runtime.CANDIDATE_GEOMETRY_FORWARD_CALLS
    if args.unbiased_physical_primary:
        from unbiased_gsplat import physical_ordering as physical_primary_runtime
        physical_primary_runtime.PHYSICAL_PRIMARY_FORWARD_CALLS=0
    if args.production:
        from unbiased_gsplat import _cuda
        if not _cuda.cuda_available():raise RuntimeError('Production hybrid requires the existing owned Surfel CUDA kernels')
        native_rasterize=_cuda.rasterize_unbiased_aux
        def counted_rasterize(*values,**keywords):
            native_calls[0]+=1
            return native_rasterize(*values,**keywords)
        _cuda.rasterize_unbiased_aux=counted_rasterize
    inputs=ColmapInputs(args.data_dir,labels=args.labels,label_subject=args.label_subject,factor=args.factor,cache_views=args.cache_views)
    args.training_view_policy='all_input_frames' if args.production else 'benchmark_holdout_eighth'
    if args.production:
        inputs.train_indices=list(range(len(inputs.rows)));inputs.test_indices=[]
    keep,hair,partition=inputs.point_partition()
    points,colors=inputs.points[keep],inputs.colors[keep];hair=hair[keep]
    if args.max_points and len(points)>args.max_points:
        select=np.random.default_rng(args.seed).choice(len(points),args.max_points,replace=False);points,colors,hair=points[select],colors[select],hair[select]
    parent=torch.load(args.resume,map_location='cpu',weights_only=False) if args.resume else None
    orientation_report={}
    orientation_keywords={'orientation_init':args.unbiased_orientation_init} if args.unbiased_orientation_init!='uniform_sphere' else {}
    if args.unbiased_orientation_init=='sfm_plane':orientation_keywords['orientation_report']=orientation_report
    if args.production and args.mode=='joint':
        if parent is not None:
            expected={'pgsr':parent['actual_optimizer_updates'],'unbiased':parent['actual_optimizer_updates']}
            strategy=parent.get('initialization_strategy',parent['config'].get('initialization_strategy'))
            if (strategy not in (None,'regional_joint_from_sfm') or parent.get('warmup_updates',0)
                    or parent['config'].get('warmup_updates',0)
                    or parent.get('updates_by_family',expected)!=expected):
                raise ValueError('Production regional training requires both families fromupdate0; staged checkpoint rejected')
        args.initialization_strategy='regional_joint_from_sfm'
    if parent is not None:
        assert parent['mode']==args.mode and parent['actual_optimizer_updates']<=args.max_steps
        if args.production and parent['config'].get('training_view_policy')!='all_input_frames':
            raise ValueError('Checkpoint uses a different or unknown training-view policy; start all-input production training from raw SfM')
        if parent.get('initialization_strategy') == 'pgsr_surface_warmstart':
            args.initialization_strategy=parent['initialization_strategy']
            args.warmup_updates=parent['warmup_updates']
        validate_normalization_resume(parent['normalization_center'],parent['normalization_radius'],inputs.center,inputs.radius)
        validate_view_batch_resume(parent['config'],args.view_batch_size,training_state_present=parent.get('training_state') is not None)
        validate_orientation_resume(parent['config'],args.unbiased_orientation_init)
        validate_surface_midpoint_resume(parent['config'],args.unbiased_surface_midpoint)
        validate_front_convergence_resume(parent['config'],args.unbiased_front_convergence)
        validate_terminal_surface_resume(parent['config'],args.unbiased_terminal_surface)
        validate_coherent_midpoint_resume(parent['config'],args.unbiased_coherent_midpoint)
        validate_input_photo_consensus_resume(parent['config'],args.unbiased_input_photo_consensus)
        validate_refinement_resume(parent['config'],args.unbiased_refinement_statistic)
        for name in ['seed','factor','max_points']:
            assert parent['config'][name]==getattr(args,name),'Resume input/config mismatch: '+name
        assert parent['config'].get('background','black')==args.background,'Resume background mismatch'
        assert parent['config'].get('label_subject','subject3')==args.label_subject,'Resume label subject mismatch'
        assert all(opacity_reset_histories_match(family,parent['actual_optimizer_updates'],
            previous_policy=parent['config'].get('opacity_reset_policy','legacy'),requested_policy=args.opacity_reset_policy)
            for family in parent['families']),'Resume opacity reset history mismatch'
        assert parent['config'].get('leakage_reduction','violators')==args.leakage_reduction,'Resume leakage reduction mismatch'
        assert parent['config'].get('pgsr_normal_weight',.015)==args.pgsr_normal_weight,'Resume PGSR normal weight mismatch'
        assert parent['config'].get('hair_family','pgsr')==args.hair_family,'Resume native semantic role mismatch'
        assert parent['config'].get('occlusion_aware_identity',False)==args.occlusion_aware_identity,'Resume identity visibility mismatch; use a new training job'
        assert parent['config'].get('joint_physical_visibility',False)==args.joint_physical_visibility,'Resume geometric visibility mismatch; use a new training job'
        assert parent['config'].get('unbiased_occlusion_aware_normal',False)==args.unbiased_occlusion_aware_normal,'Resume normal support mismatch'
        assert parent['config'].get('unbiased_dual_geometry',False)==args.unbiased_dual_geometry,'Resume dual geometry mismatch; changed geometry starts from raw SfM'
        if parent['config'].get('unbiased_reference_regularizers',False)!=args.unbiased_reference_regularizers:
            raise ValueError('Resume Surfel regularizer contract mismatch; start the changed recipe from raw SfM')
        assert parent['config'].get('unbiased_physical_primary',False)==args.unbiased_physical_primary,'Resume physical primary mismatch; changed ordering starts from raw SfM'
        assert parent['config'].get('surfel_selected_geometry',False)==args.surfel_selected_geometry,'Resume selected Surfel geometry mismatch'
        assert parent['config'].get('multiview_families','pgsr')==args.multiview_families,'Resume multi-view family mismatch'
        assert parent['config'].get('unbiased_normal_weight',.05)==args.unbiased_normal_weight,'Resume Unbiased normal weight mismatch'
        assert parent['config'].get('unbiased_convergence_weight',7.)==args.unbiased_convergence_weight,'Resume Unbiased convergence weight mismatch'
        assert parent['config'].get('unbiased_convergence_variant','adjacent')==args.unbiased_convergence_variant,'Resume Unbiased convergence variant mismatch'
        assert parent['config'].get('unbiased_start_iter',1000)==args.unbiased_start_iter,'Resume Unbiased geometry timing mismatch'
        assert parent['config'].get('unbiased_sfm_anchor_weight',0.)==args.unbiased_sfm_anchor_weight,'Resume input-SfM anchor weight mismatch'
        assert parent['config'].get('unbiased_sfm_anchor_domain','native_surfel')==args.unbiased_sfm_anchor_domain,'Resume input-SfM surface domain mismatch'
        if parent['config'].get('regional_identity_consistency',False)!=args.regional_identity_consistency:
            raise ValueError('Resume regional identity-confidence policy mismatch; start changed training from raw SfM')
        assert parent['config'].get('unbiased_reduction','mean_segments')==args.unbiased_reduction,'Resume Unbiased reduction mismatch'
        assert parent['config'].get('unbiased_auxiliary_refinement_gradient',False)==args.unbiased_auxiliary_refinement_gradient,'Resume auxiliary refinement statistic mismatch'
        if args.mode=='unbiased':
            assert parent['config'].get('single_surfel_exposure',False)==args.single_surfel_exposure,'Resume single-Surfel exposure mismatch'
        if args.mode=='joint':
            assert parent['config'].get('native_anchor_color_space','raw')==args.native_anchor_color_space,'Resume native anchor color space mismatch; use a new training job'
        validate_resume_dataset_paths(parent['config'],args.data_dir,args.labels,args.output)
        families={f:{k:torch.nn.Parameter(v.to('cuda')) for k,v in values.items()} for f,values in parent['families'].items()}
    elif args.mode=='joint':
        if not hair.any() or hair.all():raise RuntimeError('Existing segmented raw COLMAP does not contain both families')
        masks=initial_family_masks(hair,args.hair_family)
        families={family:initialize_parameters(points[masks[family]],colors[masks[family]],axes=axes,seed=args.seed,
                      **(orientation_keywords if family=='unbiased' else {}))
                  for family,axes in [('pgsr',3),('unbiased',2)]}
    else:families={args.mode:initialize_parameters(points,colors,axes=3 if args.mode=='pgsr' else 2,seed=args.seed,
                     **(orientation_keywords if args.mode=='unbiased' else {}))}
    args.unbiased_orientation_report=parent['config'].get('unbiased_orientation_report') if parent is not None else (orientation_report or None)
    exposure=torch.nn.Parameter(torch.zeros(len(inputs.rows),2,device='cuda')) if 'pgsr' in families or args.single_surfel_exposure else None
    optimizer=make_optimizer(families,exposure);pg_config=PlanarGaussianConfig(normal_weight=args.pgsr_normal_weight);mv_config=PGSRMultiViewConfig()
    if parent is not None:
        if exposure is not None:exposure.data.copy_(parent['exposure'].to('cuda'))
        optimizer.load_state_dict(parent['optimizer_state'])
    graph=build_camera_neighbor_graph(torch.tensor(np.stack([r['c2w'] for r in inputs.rows]),dtype=torch.float32),list(range(len(inputs.rows))),config=mv_config)
    # Training split only for neighbor supervision.
    training=set(inputs.train_indices);graph=[tuple(i for i in row if i in training) for row in graph]
    converter=lambda item:tensors(item,background=background if args.production else None,camera_geometry_cache=args.camera_geometry_cache,full_camera_geometry_cache=args.full_camera_geometry_cache)
    view_tensors=ViewTensorCache(inputs.load,converter,enabled=args.view_tensor_cache=='gpu',palette=args.view_cache_palette)
    view_tensors.preload(range(len(inputs.rows)))
    sfm_targets=None;sfm_target_stats=None
    if args.unbiased_sfm_anchor_weight>0:
        raw_targets,sfm_target_stats=build_training_targets(inputs)
        if sfm_target_stats['observations']==0:raise RuntimeError('No eligible original training SfM surface measurements')
        sfm_targets=[tuple(torch.as_tensor(v,device='cuda',dtype=torch.float32) for v in pair) for pair in raw_targets]
    surfel_normal_start=REFERENCE_SURFEL_NORMAL_START_STEP if reference_regularizers else args.unbiased_start_iter
    surfel_convergence_start=REFERENCE_SURFEL_CONVERGENCE_START_STEP if reference_regularizers else args.unbiased_start_iter
    geometry_starts={f:surfel_normal_start if f=='unbiased' else 1000 for f in families}
    manifest=dict(mode=args.mode,partition=partition,initial_primitives={f:len(p['means']) for f,p in families.items()},
        dataset=args.data_dir,labels=args.labels,label_subject=args.label_subject,label_directory=str(inputs.label_directory) if inputs.label_directory else None,
        expected_views=len(inputs.rows),training_views=len(inputs.train_indices),heldout_views=len(inputs.test_indices),
        training_view_policy=args.training_view_policy,source_snapshot=None if args.production else (ROOT/'SNAPSHOT_SHA256').read_text().strip(),
        raw_colmap_only=True,pretrained_parameters=False,one_adam=True,full_original_rgb=True,packed=False,
        renderers={'pgsr':'gsplat.rasterization 3-axis plane attributes','unbiased':'gsplat.rasterization_2dgs + independent corrected-depth CUDA'},
        sharing_processes_at_start=[] if args.studio else gpu_processes(),max_seconds=args.seconds,maximum_updates=args.max_steps,
        memory_fraction=args.memory_fraction,
        geometry_start=min(geometry_starts.values()),geometry_start_by_family=geometry_starts,
        multiview_start=1500,multiview_stride=4,
        multiview_neighbor_gradient='one-sided explicit performance variant',two_aggregate_layers=args.mode=='joint',
        checkpoint_author_equivalence=False,commercial_original_code_license='Apache-2.0')
    manifest.update(view_batch_size=args.view_batch_size,view_objective_reduction='mean complete original per-view losses',
                    refinement_statistic_unscale=args.view_batch_size,term_counter_scope='per-view evaluations')
    manifest['regional_appearance_vjp']='Complete RGBA derivative weighted per family; native geometry/semantic/leakage opacity unweighted'
    manifest['joint_geometry_export']='nearest valid positive native depth with own alpha>=.5; no combined-alpha borrowing or label-forced depth'
    manifest['joint_native_radiance_anchor']='0.5 weighted native L1 per family on supported identity or observed background; unlabelled/unknown foreground abstains; opposite trusted identity excluded; no current-validity gate'
    manifest['native_anchor_color_space']=args.native_anchor_color_space
    manifest['unlabelled_identity_policy']='Complete composite RGBA derivatives are neutral (1,1) on unlabelled/unknown foreground; independent native anchors abstain there; physical foreground geometry is retained'
    manifest['opacity_reset_policy']=args.opacity_reset_policy
    manifest['unbiased_orientation_init']=args.unbiased_orientation_init
    manifest['unbiased_refinement_statistic']=args.unbiased_refinement_statistic
    manifest['unbiased_surface_midpoint']=args.unbiased_surface_midpoint
    manifest['unbiased_terminal_surface']=args.unbiased_terminal_surface
    manifest['unbiased_occlusion_aware_normal']=args.unbiased_occlusion_aware_normal
    manifest['unbiased_dual_geometry']=args.unbiased_dual_geometry
    manifest['unbiased_reference_regularizers']=args.unbiased_reference_regularizers
    if 'unbiased' in families:
        manifest.update(unbiased_normal_start_step=surfel_normal_start,
                        unbiased_convergence_start_step=surfel_convergence_start)
    manifest['unbiased_physical_primary']=args.unbiased_physical_primary
    manifest['unbiased_coherent_midpoint']=args.unbiased_coherent_midpoint
    manifest['unbiased_front_convergence']=args.unbiased_front_convergence
    manifest['leakage_reduction']=args.leakage_reduction
    manifest['pgsr_normal_weight']=args.pgsr_normal_weight
    manifest.update(regional_image_cudagraph=args.regional_image_cudagraph,regional_image_fusion=args.regional_image_fusion,tensor_loss_fusion=args.tensor_loss_fusion,loss_cudagraphs=args.loss_cudagraphs,
                    separable_ssim=args.separable_ssim,normal_compile_mode='default')
    manifest['hair_family']=args.hair_family
    manifest['occlusion_aware_identity']=args.occlusion_aware_identity
    manifest['joint_physical_visibility']=args.joint_physical_visibility
    manifest['surfel_selected_geometry']=args.surfel_selected_geometry
    manifest['camera_geometry_cache']=args.camera_geometry_cache
    manifest['full_camera_geometry_cache']=args.full_camera_geometry_cache
    manifest['single_surfel_exposure']=args.single_surfel_exposure
    manifest['unbiased_sfm_anchor_weight']=args.unbiased_sfm_anchor_weight
    manifest['unbiased_orientation_report']=args.unbiased_orientation_report
    manifest['unbiased_sfm_anchor_domain']=args.unbiased_sfm_anchor_domain
    manifest['unbiased_sfm_anchor_inputs']=sfm_target_stats
    manifest['aux_depth_vjp_carry']=args.aux_depth_vjp_carry
    manifest['unbiased_auxiliary_refinement_gradient']=args.unbiased_auxiliary_refinement_gradient
    manifest['multiview_families']=args.multiview_families
    manifest.update(unbiased_normal_weight=args.unbiased_normal_weight,unbiased_convergence_weight=args.unbiased_convergence_weight,
        unbiased_reduction=args.unbiased_reduction,view_tensor_cache=args.view_tensor_cache,
        gpu_cached_views=len(view_tensors.entries),gpu_view_cache_bytes=view_tensors.bytes,
        view_cache_palette=args.view_cache_palette,gpu_palette_tensors=view_tensors.palette_tensors,
        gpu_view_cache_storage_bytes=view_tensors.storage_bytes,
        cache_semantics='Exact immutable converted original tensors; same dtype, values, view sampler and held-out split')
    manifest['active_multiview_families']=list(active_multiview_families(families,args.multiview_families))
    manifest['leakage_support']='Wrong-family visible alpha excess above0.05; trusted-support divides by all trusted pixels, violators by positive violations only'
    manifest['opacity_reset_lifetime']='Every3000 zero-based steps; refinement policy stops PGSR at15000 and Unbiased at20000; legacy continues through27000'
    manifest['source_owned_cuda']=dict(backend='owned_cuda',module=_cuda.unbiased_gsplat_cuda.__file__) if args.production else get_build_info()
    manifest.update(background_rgb=[1.,1.,1.] if args.background=='white' else [0.,0.,0.],
                    foreground_rgb_composition='Original foregroundRGB; configuredbackground outside mask' if args.production else 'ExistingpreparedRGB',
                    background_composition='once after native-family/guarded alpha composition; geometry attributes remain premultiplied',
                    host_cached_views=args.cache_views,
                    standalone_geometry_semantic_abstention=False,
                    input_cache_bytes=sum(sum(v.nbytes for v in item.values() if hasattr(v,'nbytes')) for item in inputs.cache.values()))
    manifest.update(resume_from_own_checkpoint=bool(parent),initial_optimizer_updates=parent['actual_optimizer_updates'] if parent else 0,
                    performance_target_seconds=1800 if args.production else 3600,hard_time_limit=args.seconds>0,
                    parent_checkpoint_sha256=hashlib.sha256(args.resume.read_bytes()).hexdigest() if parent and not args.production else None,
                    parent_source_snapshot=parent.get('source_snapshot') if parent else None)
    write_json(out/'RUN_MANIFEST.json',manifest)
    sums={f:torch.zeros(len(p['means']),device='cuda') for f,p in families.items()};counts={f:torch.zeros_like(sums[f]) for f in families}
    metric_file=out/'training.jsonl';updates=0;stack=[];regularizer_hits=0;mv_hits=0;mv_family_hits={f:0 for f in families}
    view_loss_evaluations=0
    sfm_anchor_calls=0
    input_consensus_calls=0
    input_consensus_pixels=torch.zeros((),device='cuda',dtype=torch.int64) if args.unbiased_input_photo_consensus else None
    identity_evidence=RegionalIdentityEvidence() if args.regional_identity_consistency else None
    sfm_anchor_observations=torch.zeros((),device='cuda',dtype=torch.int64) if sfm_targets is not None else None
    if parent is not None:
        updates=parent['actual_optimizer_updates'];saved=parent.get('training_state')
        if saved is None:
            assert all(updates>=(15000 if f=='pgsr' else 20000) for f in families),'Legacy checkpoint lacks active refinement statistics'
            saved=replay_sampler(updates,args.seed,inputs.train_indices,graph,has_pgsr='pgsr' in families)
            manifest['legacy_sampling_state_replayed']=True
        else:
            sums={f:v.to('cuda') for f,v in saved['gradient_sums'].items()}
            counts={f:v.to('cuda') for f,v in saved['observation_counts'].items()}
            manifest['legacy_sampling_state_replayed']=False
        random.setstate(saved['python_rng']);stack=list(saved['view_stack'])
        regularizer_hits=saved['regularizer_updates'];mv_hits=saved['multiview_updates']
        mv_family_hits=saved.get('multiview_family_updates',{f:mv_hits if f=='pgsr' else 0 for f in families})
        sfm_anchor_calls=saved.get('sfm_anchor_calls',0)
        input_consensus_calls=saved.get('input_photo_consensus_calls',0)
        if input_consensus_pixels is not None:input_consensus_pixels.fill_(saved.get('input_photo_consensus_pixels',0))
        if identity_evidence is not None:
            if 'regional_identity_evidence' not in saved:raise ValueError('Missing regional identity witness resume state')
            identity_evidence.load_state_dict(saved['regional_identity_evidence'],device='cuda')
        if sfm_anchor_observations is not None:sfm_anchor_observations.fill_(saved.get('sfm_anchor_observations',0))
        if 'numpy_rng' in saved:np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(parent['rng_cpu']);torch.cuda.set_rng_state_all(parent['rng_cuda'])
        write_json(out/'RUN_MANIFEST.json',manifest)
    initial_updates=updates
    def saved_state():
        return dict(python_rng=random.getstate(),numpy_rng=np.random.get_state(),view_stack=stack.copy(),
                    **(dict(regional_identity_evidence=identity_evidence.state_dict()) if identity_evidence is not None else {}),
                    regularizer_updates=regularizer_hits,multiview_updates=mv_hits,
                    multiview_family_updates=mv_family_hits.copy(),
                    sfm_anchor_calls=sfm_anchor_calls,
                    input_photo_consensus_calls=input_consensus_calls,
                    input_photo_consensus_pixels=int(input_consensus_pixels) if input_consensus_pixels is not None else 0,
                    sfm_anchor_observations=int(sfm_anchor_observations) if sfm_anchor_observations is not None else 0,
                    gradient_sums={f:v.detach().cpu() for f,v in sums.items()},
                    observation_counts={f:v.detach().cpu() for f,v in counts.items()})
    train_deadline=deadline-(0 if args.smoke else min(240,args.seconds*.22))
    from contextlib import nullcontext
    from unbiased_gsplat.surfel_execution import NativeFamilyStreams, NativeSurfelAuxiliaryStreams
    from unbiased_gsplat.shared_surfel_execution import NativeSurfelSharedForward
    from unbiased_gsplat.adjoint_tile_execution import NativeAdjointTiles
    family_scope=NativeSurfelSharedForward(render_planar_gaussians,render_unbiased_surfels,last_backward=args.surfel_last_backward) if args.shared_surfel_forward else NativeSurfelAuxiliaryStreams(render_planar_gaussians,render_unbiased_surfels) if args.parallel_surfel_auxiliary else NativeFamilyStreams(render_planar_gaussians,render_unbiased_surfels) if args.parallel_family_streams else nullcontext()
    with family_scope as family_streams, (NativeAdjointTiles() if args.zero_adjoint_tiles else nullcontext()) as adjoint_tiles, (NativePhysicalRefinement() if args.unbiased_refinement_statistic=='phase4' else nullcontext()) as refinement_scope:
        pgsr_renderer=family_streams.pgsr if family_streams is not None else render_planar_gaussians
        surfel_renderer=family_streams.surfel if family_streams is not None else render_unbiased_surfels
        for step in range(updates,args.max_steps):
            if refinement_scope is not None:refinement_scope.enabled=population_statistics_needed('unbiased',step)
            if step%100==0:production_disk_check(args.production and not args.studio)
            if time.time()>=train_deadline:break
            if args.loss_cudagraphs:torch.compiler.cudagraph_mark_step_begin()
            step_loss=None;step_components={};batch_indices=[]
            for batch_index in range(args.view_batch_size):
                if not stack:stack=inputs.train_indices.copy();random.shuffle(stack)
                index=stack.pop();original_item=view_tensors.load(index);degree=min(3,step//1000)
                item=identity_evidence.apply(index,original_item) if identity_evidence is not None else original_item
                batch_indices.append(index)
                prediction,layers=render(families,item,degree,background=background,leakage_reduction=args.leakage_reduction,hair_family=args.hair_family,
                                         occlusion_aware_identity=args.occlusion_aware_identity,unbiased_reduction=args.unbiased_reduction,joint_physical_visibility=args.joint_physical_visibility,surfel_selected_geometry=args.surfel_selected_geometry,
                                         defer_native_anchors=args.mode=='joint' and args.native_anchor_color_space=='exposure',
                                         unbiased_convergence_variant=args.unbiased_convergence_variant,
                                         unbiased_auxiliary_refinement_gradient=args.unbiased_auxiliary_refinement_gradient,
                                         unbiased_surface_midpoint=args.unbiased_surface_midpoint,unbiased_terminal_surface=args.unbiased_terminal_surface,
                                         unbiased_coherent_midpoint=args.unbiased_coherent_midpoint,unbiased_occlusion_aware_normal=args.unbiased_occlusion_aware_normal,unbiased_physical_primary=args.unbiased_physical_primary,unbiased_dual_geometry=args.unbiased_dual_geometry,
                                         unbiased_reference_regularizers=args.unbiased_reference_regularizers,
                                         unbiased_front_convergence=args.unbiased_front_convergence,
                                         # ColmapInputs already divides positions and camera translations by radius.
                                         # Raw SfM units are arbitrary and must not rescale this objective again.
                                         unbiased_convergence_physical_scale=1.,
                                         regional_image_operator=regional_image_operator,
                                         regional_exposure=exposure[index] if regional_image_enabled else None,family_streams=family_streams)
                for family,layer in layers.items():
                    if not population_statistics_needed(family,step):continue
                    for key in ['means2d','gradient_2dgs']:
                        tensor=layer['meta'].get(key)
                        if tensor is not None and tensor.requires_grad:tensor.retain_grad()
                if regional_image_enabled:
                    color=prediction['_regional_color'];effective_exposure=prediction['_regional_effective_exposure']
                else:color,effective_exposure=photo_operator(prediction['rgb'],item['rgb'],exposure[index] if exposure is not None else None,return_exposure=True)
                if args.mode=='joint' and args.native_anchor_color_space=='exposure' and not regional_image_enabled:
                    for family in ['pgsr','unbiased']:
                        prediction['native_'+family+'_anchor']=native_radiance_anchor(layers[family],item['rgb'],prediction['native_anchor_weights'][family],
                            background=background,exposure=effective_exposure)
                loss=color;components={'color':color}
                if 'pgsr' in layers:
                    flatten=pg_config.flatten_weight*layers['pgsr']['flatten_loss'];loss=loss+flatten;components['flatten']=flatten
                pgsr_geometry='pgsr' in layers and geometry_regularizers_active('pgsr',step,smoke=args.smoke)
                surfel_convergence,surfel_normal=surfel_regularizer_activity(step,reference_regularizers=reference_regularizers,
                    unbiased_start_iter=args.unbiased_start_iter,smoke=args.smoke)
                surfel_geometry='unbiased' in layers and (surfel_convergence or surfel_normal)
                if pgsr_geometry or surfel_geometry:
                    regularizer_hits+=1
                    if pgsr_geometry:
                        layer=layers['pgsr'];support=physical_geometry_domain(item['foreground'])
                        normal_keywords={}
                        if 'geometry_visibility' in layer:normal_keywords['gradient_visibility']=normal_stencil_visibility(layer['geometry_visibility'])
                        normal=normal_operator(layer['depth'],layer['valid'],layer['normal_sum_camera'],item['rgb'],item['c2w'],item['K'],support,**normal_keywords)
                        loss=loss+pg_config.normal_weight*normal;components['pgsr_normal']=normal
                    if surfel_geometry:
                        layer=layers['unbiased'];convergence_term,normal_term=unbiased_regularizer_terms(layer['convergence_loss'],layer['normal_loss'],normal_weight=args.unbiased_normal_weight,convergence_weight=args.unbiased_convergence_weight)
                        if surfel_convergence:
                            loss=loss+convergence_term;components['convergence']=layer['convergence_loss']
                        if surfel_normal:
                            loss=loss+normal_term;components['unbiased_normal']=layer['normal_loss']
                if sfm_targets is not None:
                    measurement_depth,measurement_valid=surface_anchor_fields(layers,domain=args.unbiased_sfm_anchor_domain,hair_family=args.hair_family)
                    sfm_loss,sfm_count=surface_anchor_loss(measurement_depth,measurement_valid,*sfm_targets[index])
                    loss=loss+args.unbiased_sfm_anchor_weight*sfm_loss
                    sfm_anchor_calls+=1;sfm_anchor_observations+=sfm_count.detach()
                    components['sfm_surface_anchor']=sfm_loss;components['sfm_surface_measurements']=sfm_count
                if args.mode=='joint':
                    loss=loss+.03*prediction['semantic_loss']+prediction['leakage_loss']+.5*(prediction['native_pgsr_anchor']+prediction['native_unbiased_anchor'])
                    components['semantic']=prediction['semantic_loss'];components['leakage']=prediction['leakage_loss']
                    components['native_pgsr_anchor']=prediction['native_pgsr_anchor'];components['native_unbiased_anchor']=prediction['native_unbiased_anchor']
                mv_families=active_multiview_families(layers,args.multiview_families)
                if mv_families and graph[index] and ((step>=1500 and step%4==0) or args.smoke):
                    neighbor_index=random.choice(graph[index]);neighbor=view_tensors.load(neighbor_index)
                    neighbor_layers={}
                    with torch.no_grad():
                        neighbor_families=tuple(layers) if args.joint_physical_visibility and len(layers)==2 else mv_families
                        for family in neighbor_families:
                            if family=='pgsr':neighbor_layers[family]=pgsr_renderer(families[family],neighbor['c2w'],neighbor['K'],neighbor['rgb'].shape[1],neighbor['rgb'].shape[0],degree,
                                world_to_camera=neighbor.get('world_to_camera'),intrinsics_inverse=neighbor.get('intrinsics_inverse'))
                            else:neighbor_layers[family]=surfel_renderer(families[family],neighbor['c2w'],neighbor['K'],neighbor['rgb'].shape[1],neighbor['rgb'].shape[0],degree,auxiliary=False,
                                world_to_camera=neighbor.get('world_to_camera'),
                                auxiliary_camera_to_world=neighbor.get('auxiliary_camera_to_world'),
                                primary_camera_to_world=neighbor.get('primary_camera_to_world'),
                                **(dict(surface_midpoint=True) if args.unbiased_surface_midpoint else {}),
                                **(dict(terminal_surface=True) if args.unbiased_terminal_surface else {}),
                                **(dict(physical_primary=True) if args.unbiased_physical_primary else {}),
                                **(dict(dual_geometry=True,front_convergence=args.unbiased_front_convergence,convergence_physical_scale=1.) if args.unbiased_dual_geometry else {}))
                    if identity_evidence is not None:
                        conflicts=visible_identity_conflicts(layers,neighbor_layers,original_item,neighbor,
                            depth_tolerance=args.tsdf_voxel,reprojection_tolerance=mv_config.reprojection_threshold_pixels,hair_family=args.hair_family)
                        identity_evidence.observe(index,graph[index].index(neighbor_index),conflicts)
                        del conflicts
                    mv_loss,mv_results=native_multiview_constraints(layers,neighbor_layers,item,neighbor,policy=args.multiview_families,config=mv_config,
                        joint_physical_visibility=args.joint_physical_visibility,hair_family=args.hair_family)
                    loss=loss+mv_loss;mv_hits+=1
                    for family,consistency in mv_results.items():
                        mv_family_hits[family]+=1
                        components[family+'_mv_geometry']=consistency.geometry_loss;components[family+'_mv_photo']=consistency.photometric_loss
                        components[family+'_mv_geometry_pixels']=consistency.geometry_denominator;components[family+'_mv_photo_pixels']=consistency.photometric_denominator
                    del neighbor,neighbor_layers,mv_results,mv_loss,consistency
                    if args.unbiased_input_photo_consensus:
                        input_neighbors=[view_tensors.load(j) for j in graph[index]]
                        input_photo=input_photometric_consensus(layers['unbiased'],original_item,input_neighbors,config=mv_config)
                        loss=loss+mv_config.photometric_weight*input_photo['loss']
                        input_consensus_calls+=1;input_consensus_pixels+=input_photo['denominator'].detach().to(torch.int64)
                        components['input_photo_consensus']=input_photo['loss']
                        components['input_photo_consensus_pixels']=input_photo['denominator']
                        del input_neighbors,input_photo
                batch_loss_for_backward(loss,args.view_batch_size).backward();view_loss_evaluations+=1
                try:validate_adam_inputs(loss,families,exposure)
                except RuntimeError:
                    checkpoint(out/'FAILED_PRE_ADAM.pt',families,optimizer,exposure,updates,inputs,args)
                    raise
                for family,params in families.items():
                    if not population_statistics_needed(family,step):continue
                    meta=layers[family]['meta'];screen=meta.get('gradient_2dgs',meta.get('means2d'));gradient=screen.grad if screen is not None else None
                    if gradient is not None:
                        visible=meta['radii'][0]
                        if visible.ndim==2:visible=visible.max(-1).values
                        visible=visible>0
                        if family=='unbiased' and args.unbiased_refinement_statistic=='phase4':
                            demand=phase4_view_statistic(screen,args.view_batch_size)
                        else:
                            g=unscale_view_gradient(gradient[0].detach(),args.view_batch_size);g[...,0]*=item['rgb'].shape[1]/2;g[...,1]*=item['rgb'].shape[0]/2
                            demand=torch.linalg.vector_norm(g[...,:2],dim=-1)
                        sums[family]+=demand*visible;counts[family]+=visible
                    del meta,screen,gradient
                value=batch_loss_for_backward(loss.detach(),args.view_batch_size)
                step_loss=value if step_loss is None else step_loss+value
                for key,value in components.items():
                    value=batch_loss_for_backward(value.detach().to(loss.dtype),args.view_batch_size)
                    step_components[key]=step_components[key]+value if key in step_components else value
                del item,prediction,layers,loss,color,components,layer
            update_mean_learning_rate(optimizer,step);adam_step(optimizer,fused_groups=args.joint_group_adam);updates+=1;optimizer.zero_grad(set_to_none=True)
            if not args.smoke:
                if step>=500 and step%100==0:
                    for family,params in families.items():
                        stop=15000 if family=='pgsr' else 20000
                        if step<stop:
                            refinement=refine_population(params,optimizer,family,sums[family],counts[family],max_points=args.max_points)
                            sums[family]=torch.zeros(len(params['means']),device='cuda');counts[family]=torch.zeros_like(sums[family])
                            print(json.dumps({'phase':'refine','mode':args.mode,'step':updates,'family':family,**refinement}),flush=True)
                if step>0 and step%3000==0:
                    for family,params in families.items():
                        if reset_opacities_if_due(params,optimizer,family,step,policy=args.opacity_reset_policy):
                            print(json.dumps({'phase':'opacity_reset','mode':args.mode,'family':family,'zero_based_step':step,'actual_optimizer_updates':updates,'policy':args.opacity_reset_policy}),flush=True)
                if step>=20000 and 'unbiased' in families:
                    with torch.no_grad():families['unbiased']['scales'].clamp_max_(np.log(.01))
            if step%100==0 or args.smoke:
                row=dict(step=updates,elapsed_seconds=time.time()-start,loss=float(step_loss),primitives={f:len(p['means']) for f,p in families.items()},
                         components={k:float(v) for k,v in step_components.items()},view_batch_size=args.view_batch_size,batch_views=batch_indices,
                         peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
                if identity_evidence is not None:row['regional_identity_uncertain_pixels']=identity_evidence.uncertain_pixels
                with metric_file.open('a') as stream:stream.write(json.dumps(row)+'\n')
                write_json(out/'PROGRESS.json',row);print(json.dumps(row),flush=True)
            del step_loss,step_components
            if not args.smoke and args.save_interval>0 and updates%args.save_interval==0:
                checkpoint(out/'resume_latest.pt',families,optimizer,exposure,updates,inputs,args,saved_state())
    if args.loss_cudagraphs:
        CUDAGraphNode.run_graph=original_run_graph
        skips=int(compiler_counters['inductor'].get('cudagraph_skips',0))-skips_before
        if updates>initial_updates and (graph_replays==0 or ssim_forward_replays==0 or skips):
            raise RuntimeError('Requested regional SSIM graph did not replay without skips')
    torch.cuda.synchronize();training_seconds=time.time()-start
    checkpoint(out/'checkpoint.pt',families,optimizer,exposure,updates,inputs,args,saved_state())
    geometry=None
    if not args.smoke and not args.no_export:geometry=export_and_evaluate(families,inputs,min(3,max(0,(updates-1)//1000)),out,deadline,args.mode,background,
                                               voxel_length=args.tsdf_voxel,sdf_trunc=args.tsdf_trunc,hair_family=args.hair_family,production=args.production,
                                               export_ray_depth_order=args.export_ray_depth_order,unbiased_surface_midpoint=args.unbiased_surface_midpoint,unbiased_terminal_surface=args.unbiased_terminal_surface,unbiased_physical_primary=args.unbiased_physical_primary,unbiased_dual_geometry=args.unbiased_dual_geometry,studio=args.studio)
    summary=dict(status='TRAINING_COMPLETE' if updates==args.max_steps else 'TRAINING_PREFIX_TIME_LIMIT',mode=args.mode,actual_optimizer_updates=updates,max_points=args.max_points,
                 training_view_policy=args.training_view_policy,training_views=len(inputs.train_indices),heldout_views=len(inputs.test_indices),
                 requested_updates=args.max_steps,elapsed_seconds=time.time()-start,regularizer_updates=regularizer_hits,multiview_updates=mv_hits,
                 peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,checkpoint_sha256=None if args.production else hashlib.sha256((out/'checkpoint.pt').read_bytes()).hexdigest(),
                 sharing_processes_at_end=[] if args.studio else gpu_processes(),geometry=geometry,geometry_accepted=False)
    summary.update(**staged_updates(args,updates))
    summary.update(view_batch_size=args.view_batch_size,view_loss_evaluations_this_process=view_loss_evaluations,
                   term_counter_scope='per-view evaluations')
    summary.update(full_camera_geometry_cache=args.full_camera_geometry_cache,
                   cached_primary_calibration_calls=dict(cached_calibration.CALLS))
    summary.update(surfel_last_backward=args.surfel_last_backward)
    summary.update(zero_adjoint_tiles=args.zero_adjoint_tiles,
                   zero_adjoint_tile_calls=dict(adjoint_tiles.calls) if args.zero_adjoint_tiles else {})
    summary.update(shared_surfel_forward=args.shared_surfel_forward,
                   shared_surfel_forward_calls=dict(family_streams.calls) if args.shared_surfel_forward else {})
    summary.update(parallel_surfel_auxiliary=args.parallel_surfel_auxiliary,
                   parallel_surfel_auxiliary_calls=dict(family_streams.calls) if args.parallel_surfel_auxiliary else {})
    summary.update(parallel_family_streams=args.parallel_family_streams,
                   parallel_family_stream_calls=dict(family_streams.calls) if args.parallel_family_streams else {})
    summary.update(initial_optimizer_updates=initial_updates,updates_this_process=updates-initial_updates,
                   parent_checkpoint_sha256=manifest['parent_checkpoint_sha256'],hard_time_limit=args.seconds>0,
                   multiview_family_updates=mv_family_hits,
                   multiview_families=args.multiview_families,
                   performance_target_seconds=1800 if args.production else 3600)
    if args.production:
        physical_calls=physical_primary_runtime.PHYSICAL_PRIMARY_FORWARD_CALLS if physical_primary_runtime is not None else 0
        candidate_calls=(dual_geometry_runtime.CANDIDATE_GEOMETRY_FORWARD_CALLS-dual_geometry_calls_before
                         if dual_geometry_runtime is not None else 0)
        summary.update(training_seconds=training_seconds,native_surfel_calls=native_calls[0]+physical_calls+candidate_calls,
                       native_surfel_legacy_aux_calls=native_calls[0],native_surfel_physical_primary_calls=physical_calls,
                       native_surfel_dual_geometry_calls=candidate_calls,
                       families_axes={name:values['scales'].shape[1] for name,values in families.items()},
                       hair_family=args.hair_family,production=True)
    if 'unbiased' in families:
        summary.update(unbiased_normal_start_step=surfel_normal_start,
                       unbiased_convergence_start_step=surfel_convergence_start)
    summary.update(opacity_reset_policy=args.opacity_reset_policy,leakage_reduction=args.leakage_reduction,unbiased_convergence_weight=args.unbiased_convergence_weight,
                   unbiased_convergence_variant=args.unbiased_convergence_variant,
                   unbiased_start_iter=args.unbiased_start_iter,
                   unbiased_reference_regularizers=args.unbiased_reference_regularizers,
                   unbiased_orientation_init=args.unbiased_orientation_init,
                   unbiased_surface_midpoint=args.unbiased_surface_midpoint,unbiased_terminal_surface=args.unbiased_terminal_surface,
                   unbiased_coherent_midpoint=args.unbiased_coherent_midpoint,unbiased_occlusion_aware_normal=args.unbiased_occlusion_aware_normal,unbiased_physical_primary=args.unbiased_physical_primary,unbiased_dual_geometry=args.unbiased_dual_geometry,
                   unbiased_input_photo_consensus=args.unbiased_input_photo_consensus,
                   input_photo_consensus_calls=input_consensus_calls,
                   input_photo_consensus_pixels=int(input_consensus_pixels) if input_consensus_pixels is not None else 0,
                   unbiased_front_convergence=args.unbiased_front_convergence,
                   unbiased_convergence_units=args.unbiased_convergence_units,
                   unbiased_sfm_anchor_weight=args.unbiased_sfm_anchor_weight,unbiased_sfm_anchor_domain=args.unbiased_sfm_anchor_domain,sfm_anchor_calls=sfm_anchor_calls,
                   unbiased_orientation_report=args.unbiased_orientation_report,
                   regional_identity_consistency=args.regional_identity_consistency,
                   regional_identity_calls=identity_evidence.calls if identity_evidence is not None else 0,
                   regional_identity_uncertain_pixels=identity_evidence.uncertain_pixels if identity_evidence is not None else 0,
                   sfm_anchor_observations=int(sfm_anchor_observations) if sfm_anchor_observations is not None else 0,
                   native_anchor_color_space=args.native_anchor_color_space,
                   unbiased_normal_weight=args.unbiased_normal_weight,unbiased_reduction=args.unbiased_reduction,
                   unbiased_refinement_statistic=args.unbiased_refinement_statistic,
                   physical_phase4_backward_calls=refinement_scope.calls if refinement_scope is not None else 0,
                   joint_physical_visibility=args.joint_physical_visibility,surfel_selected_geometry=args.surfel_selected_geometry,
                   aux_depth_vjp_carry=args.aux_depth_vjp_carry,aux_depth_vjp_carry_calls=_cuda.AUX_DEPTH_VJP_CARRY_CALLS,
                   aux_front_prefix_replay=args.aux_front_prefix_replay,aux_front_prefix_replay_calls=_cuda.AUX_FRONT_PREFIX_REPLAY_CALLS,
                   aux_front_prefix_forward=args.aux_front_prefix_forward,aux_front_prefix_forward_calls=_cuda.AUX_FRONT_PREFIX_FORWARD_CALLS,
                   native_point_normals=args.native_point_normals,native_point_normal_calls=depth_normal.NATIVE_POINT_NORMAL_CALLS,
                   joint_group_adam=args.joint_group_adam,joint_group_adam_calls=optimizer_execution.GROUP_KERNEL_CALLS,
                   original_adam_initialization_calls=optimizer_execution.ORIGINAL_INITIALIZATION_CALLS,
                   regional_image_fusion=args.regional_image_fusion,regional_image_calls=regional_image_operator.calls if args.regional_image_cudagraph else regional_image_calls,
                   regional_image_cudagraph=args.regional_image_cudagraph,
                   regional_image_graph_captures=regional_image_operator.captures if args.regional_image_cudagraph else 0,
                   regional_image_graph_backward_calls=regional_image_operator.backward_calls if args.regional_image_cudagraph else 0,
                   tensor_loss_fusion=args.tensor_loss_fusion,loss_cudagraphs=args.loss_cudagraphs,
                   unbiased_auxiliary_refinement_gradient=args.unbiased_auxiliary_refinement_gradient,
                   occlusion_aware_identity=args.occlusion_aware_identity,camera_geometry_cache=args.camera_geometry_cache,
                   separable_ssim=args.separable_ssim,fusion_generated_kernels=int(fusion_metrics.generated_kernel_count)-kernels_before,
                   compiled_loss_calls=loss_calls,ssim_cuda_graph_replays=graph_replays,
                   ssim_forward_graph_replays=ssim_forward_replays)
    if consensus_before is not None:
        from unbiased_gsplat.ray_consensus import NATIVE_CALLS
        summary['ray_consensus_native_calls']={key:NATIVE_CALLS[key]-consensus_before[key] for key in NATIVE_CALLS}
    write_json(out/'RESULT.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        traceback.print_exc();raise SystemExit(1)
