"""Standalone geometry API above SplatForge's neutral runtime."""
import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from .finishing import FinishConfig, finish_mesh
from .formats import read_mesh, write_mesh
from .processing import audit, GeometryError


def main(argv=None):
    p=argparse.ArgumentParser(description='Audit or finish triangle meshes with the owned CPU geometry provider')
    p.add_argument('input',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--audit-only',action='store_true')
    p.add_argument('--cut-offset',type=float)
    p.add_argument('--cut-normal',nargs=3,type=float,default=(0.,0.,1.))
    p.add_argument('--target-faces',type=int)
    p.add_argument('--keep-all-components',action='store_true')
    p.add_argument('--clean-faces',action='store_true')
    p.add_argument('--split-fans',action='store_true')
    p.add_argument('--no-repair',action='store_true')
    p.add_argument('--repair-face-budget',type=int,default=10000)
    p.add_argument('--hole-method',choices=('projected','minimal_area'),default='projected')
    p.add_argument('--max-hole-edges',type=int)
    p.add_argument('--clip-backend',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--intersection-backend',choices=('cpu','cuda_filtered_exact'),default='cpu')
    p.add_argument('--recipe',choices=('default','native-local'),default='default')
    p.add_argument('--pre-cut-offset',type=float,default=0.)
    args=p.parse_args(argv)
    if not args.audit_only and args.output is None:p.error('--output is required for finishing')
    if args.output is not None and (args.output.resolve()==args.input.resolve() or args.output.exists()):
        p.error('choose a new output path; input and previous outputs are preserved')
    m=read_mesh(args.input)
    if args.audit_only:
        report=audit(m,intersection_backend=args.intersection_backend);print(json.dumps(report,indent=2));return 0 if report['passed'] else 1
    config=FinishConfig(cut_normal=tuple(args.cut_normal),cut_offset=args.cut_offset,keep_largest=not args.keep_all_components,
        repair=not args.no_repair,repair_face_budget=args.repair_face_budget,target_faces=args.target_faces,
        clean_invalid_faces=args.clean_faces,split_disconnected_fans=args.split_fans,
        hole_method=args.hole_method,max_hole_edges=args.max_hole_edges,
        clip_backend=args.clip_backend,intersection_backend=args.intersection_backend)
    if args.recipe == 'native-local':
        if args.no_repair or args.keep_all_components:
            p.error('native-local requires repair and a single largest component')
        config=FinishConfig.native_local(initial_cut=args.pre_cut_offset,
            final_cut=.02 if args.cut_offset is None else args.cut_offset,
            target_faces=300000 if args.target_faces is None else args.target_faces,
            clip_backend=args.clip_backend,intersection_backend=args.intersection_backend)
        config=replace(config,cut_normal=tuple(args.cut_normal),
            repair_face_budget=args.repair_face_budget,clean_invalid_faces=args.clean_faces,
            max_hole_edges=2048 if args.max_hole_edges is None else args.max_hole_edges)
    try:
        result,report=finish_mesh(m,config)
    except GeometryError as exc:
        p.exit(1,str(exc)+'\n')
    write_mesh(args.output,result)
    actual=audit(read_mesh(args.output),intersection_backend=args.intersection_backend)
    report['actual_file_audit']=actual;report['config']=asdict(config)
    args.output.with_suffix(args.output.suffix+'.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    return 0 if actual['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
