"""Bounded CPU mesh audit/finishing benchmark; never overwrites a source mesh."""
from __future__ import annotations
import argparse
import json
import platform
import sys
import time
from pathlib import Path
import numpy as np

from splatforge.mesh import FinishConfig, audit, finish_mesh
from splatforge.mesh.formats import read_mesh, write_mesh


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cut',type=float)
    p.add_argument('--target-faces',type=int)
    p.add_argument('--finish',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    record=args.output/'benchmark.json'
    if record.exists():raise FileExistsError('choose a new output directory; preserve recorded evidence')
    result={'platform':platform.platform(),'processor':platform.processor(),'python':sys.version,'numpy':np.__version__,'cases':[]}
    for source in args.input:
        tick=time.perf_counter();mesh=read_mesh(source)
        case={'source':str(source.resolve()),'read_seconds':time.perf_counter()-tick}
        try:
            if args.finish:
                mesh,case['finishing']=finish_mesh(mesh,FinishConfig(cut_offset=args.cut,target_faces=args.target_faces))
                output=args.output/(source.stem+'_forge.ply')
                if output.resolve()==source.resolve() or output.exists():raise FileExistsError(output)
                write_mesh(output,mesh);reloaded=read_mesh(output)
                case['actual_file_audit']=audit(reloaded)
                case['output']=str(output.resolve())
                case['passed']=case['actual_file_audit']['passed']
            else:
                case['audit']=audit(mesh);case['passed']=case['audit']['passed']
        except Exception as exc:
            case['passed']=False;case['error']=f'{type(exc).__name__}: {exc}'
        case['wall_seconds']=time.perf_counter()-tick
        result['cases'].append(case);record.write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(case),flush=True)
    return 0 if all(c['passed'] for c in result['cases']) else 1


if __name__=='__main__':raise SystemExit(main())
