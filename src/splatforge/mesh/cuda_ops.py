"""Owned binary64 CUDA clipping kernels with explicit CPU topology assembly."""
from functools import lru_cache
from pathlib import Path
import time
import numpy as np
from .processing import _mesh, _plane, cap_boundaries, _count_operation


@lru_cache(maxsize=8)
def _kernels(device):
    import cupy as cp
    with cp.cuda.Device(device):
        module = cp.RawModule(path=str(Path(__file__).with_name('native')/'clip.ptx'))
        return tuple(module.get_function(name) for name in ('distances','classify','intersections'))


def clip_plane_cuda(mesh, normal=(0.,0.,1.), offset=0., *, cap=True, device=0):
    """CUDA distance/classification/interpolation, CPU indexed assembly and cap.

    The PTX is owned source loaded by the CUDA driver; no runtime compiler or
    CUDA Toolkit is needed. No dtype quantization, voxel resampling or vertex
    welding. All data transfers and host work are included in total_seconds.
    """
    try:
        import cupy as cp
    except ImportError as exc:
        raise ImportError('CUDA clipping requires the geometry-cuda12 extra') from exc
    normal, offset = _plane(normal, offset)
    if max(len(mesh.vertices),len(mesh.faces)) >= 2**32:
        raise ValueError('CUDA clipping supports fewer than 2^32 vertices/faces')
    started=time.perf_counter(); kernel_seconds=0.
    with cp.cuda.Device(device):
        distance_kernel, classify_kernel, edge_kernel = _kernels(device)
        v, f = cp.asarray(mesh.vertices), cp.asarray(mesh.faces)
        distance = cp.empty(len(v),cp.float64)
        code = cp.empty(len(f),cp.int32)
        begin,end=cp.cuda.Event(),cp.cuda.Event()
        begin.record()
        if len(v):
            distance_kernel(((len(v)+127)//128,),(128,),(v,distance,np.uint32(len(v)),*map(np.float64,normal),np.float64(offset)))
        if len(f):
            classify_kernel(((len(f)+127)//128,),(128,),(f,distance,code,np.uint32(len(f))))
        end.record();end.synchronize()
        kernel_seconds+=cp.cuda.get_elapsed_time(begin,end)/1000.
        distance_host, codes = cp.asnumpy(distance),cp.asnumpy(code)
        kept=mesh.faces[codes==7]
        crossing=mesh.faces[(codes>0)&(codes<7)]
        vertices=mesh.vertices
        if len(crossing):
            starts=crossing;ends=np.roll(crossing,-1,axis=1)
            da,db=distance_host[starts],distance_host[ends]
            cuts=((da<0)&(db>0))|((da>0)&(db<0))
            edges=np.sort(np.stack((starts[cuts],ends[cuts]),axis=1),axis=1)
            unique,inverse=np.unique(edges,axis=0,return_inverse=True)
            added_gpu=cp.empty((len(unique),3),cp.float64)
            if len(unique):
                edge_gpu=cp.asarray(unique)
                begin.record()
                edge_kernel(((len(unique)+127)//128,),(128,),(v,edge_gpu,distance,added_gpu,np.uint32(len(unique))))
                end.record();end.synchronize()
                kernel_seconds+=cp.cuda.get_elapsed_time(begin,end)/1000.
            added=cp.asnumpy(added_gpu)
            axis=int(np.argmax(np.abs(normal)));other=[i for i in range(3) if i!=axis]
            added[:,axis]=(offset-(added[:,other]*normal[other]).sum(axis=1))/normal[axis]
            slot_ids=np.full((len(crossing),6),-1,np.int64)
            emit=np.zeros((len(crossing),6),bool)
            slot_ids[:,::2]=starts;emit[:,::2]=da>=0
            cut_ids=np.full_like(starts,-1);cut_ids[cuts]=inverse+len(v)
            slot_ids[:,1::2]=cut_ids;emit[:,1::2]=cuts
            positions=np.cumsum(emit,axis=1)-1
            polygon=np.full((len(crossing),4),-1,np.int64)
            rows=np.broadcast_to(np.arange(len(crossing))[:,None],emit.shape)
            polygon[rows[emit],positions[emit]]=slot_ids[emit]
            counts=emit.sum(axis=1)
            triangles=np.concatenate((kept,polygon[counts>=3][:,[0,1,2]],polygon[counts==4][:,[0,2,3]]))
            vertices=np.concatenate((vertices,added))
        else:
            triangles=kept
        clipped=_mesh(vertices,triangles)
    clip_seconds=time.perf_counter()-started
    if len(mesh.vertices) or len(mesh.faces):
        _count_operation('cuda_clip')
    result=cap_boundaries(clipped,normal,offset) if cap and len(clipped.faces) else clipped
    return result, {'cuda_clip_and_transfer_seconds':clip_seconds,
        'gpu_kernel_seconds':kernel_seconds,'cpu_cap_seconds':time.perf_counter()-started-clip_seconds,
        'total_seconds':time.perf_counter()-started}


@lru_cache(maxsize=8)
def _separation_kernel(device):
    import cupy as cp
    with cp.cuda.Device(device):
        return cp.RawModule(path=str(Path(__file__).with_name('native')/'separate.ptx')).get_function('filter_pairs')


def self_intersections_cuda(mesh, *, max_pairs=0, device=0, pair_budget=20000000):
    """GPU certified separation followed by the unchanged exact CPU classifier.

    No pair matrix, threshold relaxation or CPU-query fallback. Exceeding the
    bounded AABB candidate allocation raises instead of truncating acceptance.
    """
    import cupy as cp
    from .processing import _native
    started=time.perf_counter(); native=_native()
    pairs=native.broad_pairs(mesh.vertices,mesh.faces,pair_budget)
    kernel_seconds=0.
    with cp.cuda.Device(device):
        kernel=_separation_kernel(device)
        if len(pairs):
            v,f,p=cp.asarray(mesh.vertices),cp.asarray(mesh.faces),cp.asarray(pairs)
            keep=cp.empty(len(pairs),cp.uint8); begin,end=cp.cuda.Event(),cp.cuda.Event()
            begin.record();kernel(((len(pairs)+127)//128,),(128,),(v,f,p,keep,np.uint64(len(pairs))))
            end.record();end.synchronize();kernel_seconds=cp.cuda.get_elapsed_time(begin,end)/1000
            retained=np.ascontiguousarray(pairs[cp.asnumpy(keep).astype(bool)])
        else:
            retained=pairs
    result=native.exact_pairs(mesh.vertices,mesh.faces,retained,max_pairs)
    if len(pairs): _count_operation('self_intersections_cuda_filtered')
    result.update(aabb_pairs=len(pairs),gpu_pruned_pairs=len(pairs)-len(retained),
                  gpu_kernel_seconds=kernel_seconds,total_seconds=time.perf_counter()-started)
    return result
