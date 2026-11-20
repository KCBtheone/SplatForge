"""Explicit native FP32 raw-parameter geometry; opt-in until qualification."""
import os
import importlib.util
from pathlib import Path
import torch
from torch.autograd.function import once_differentiable

_extension=None
_extension64=None
_extension64_coordinates=None
activation_counts=dict(forward=0,backward=0)


def load_prebuilt(path,*,internal_fp64=False,coordinate_pullback=False):
    """Use an explicitly built owned binary; scientific runs do not rebuild."""
    global _extension,_extension64,_extension64_coordinates
    if coordinate_pullback and not internal_fp64:raise ValueError('Coordinate pullback requires approved internal FP64')
    name='sf_radegs_tangent_fields64_coordinates' if coordinate_pullback else ('sf_radegs_tangent_fields64' if internal_fp64 else 'sf_radegs_tangent_fields')
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if coordinate_pullback:_extension64_coordinates=module
    elif internal_fp64:_extension64=module
    else:_extension=module
    return module


def extension(internal_fp64=False,coordinate_pullback=False):
    global _extension,_extension64,_extension64_coordinates
    if coordinate_pullback and not internal_fp64:raise ValueError('Coordinate pullback requires approved internal FP64')
    current=_extension64_coordinates if coordinate_pullback else (_extension64 if internal_fp64 else _extension)
    if current is None:
        from torch.utils.cpp_extension import load
        root=os.environ.get('TORCH_EXTENSIONS_DIR')
        if not root:raise RuntimeError('Task-owned TORCH_EXTENSIONS_DIR is required')
        suffix='radegs_tangent_fields64_coordinates' if coordinate_pullback else ('radegs_tangent_fields64' if internal_fp64 else 'radegs_tangent_fields')
        from .._paths import output_path

        build = output_path(Path(root) / suffix)
        build.mkdir(parents=True,exist_ok=True)
        current=load(name='sf_'+suffix,sources=[str(Path(__file__).with_name('radegs_tangent_cuda.cu'))],
            build_directory=str(build),extra_cflags=['-O3'],
            extra_cuda_cflags=['-O3']+(['-DSF_GEOMETRY64'] if internal_fp64 else [])
                +(['-DSF_COORDINATE_PULLBACK'] if coordinate_pullback else []),verbose=False)
        if coordinate_pullback:_extension64_coordinates=current
        elif internal_fp64:_extension64=current
        else:_extension=current
    return current


class _Fields(torch.autograd.Function):
    @staticmethod
    def forward(ctx,means,quats,scales,view,k,width,height):
        ctx.save_for_backward(means,quats,scales,view,k);ctx.size=(width,height)
        activation_counts['forward']+=1
        return extension().forward(means,quats,scales,view,k,width,height)

    @staticmethod
    @once_differentiable
    def backward(ctx,gradient):
        result=extension().backward(*ctx.saved_tensors,gradient.contiguous(),*ctx.size)
        activation_counts['backward']+=1
        return *result,None,None,None,None


class _Fields64(torch.autograd.Function):
    @staticmethod
    def forward(ctx,means,quats,scales,view,k,width,height):
        ctx.save_for_backward(means,quats,scales,view,k);ctx.size=(width,height)
        activation_counts['forward']+=1
        return extension(True).forward(means,quats,scales,view,k,width,height)

    @staticmethod
    @once_differentiable
    def backward(ctx,gradient):
        result=extension(True).backward(*ctx.saved_tensors,gradient.contiguous(),*ctx.size)
        activation_counts['backward']+=1
        return *result,None,None,None,None


class _Fields64Coordinates(torch.autograd.Function):
    @staticmethod
    def forward(ctx,means,quats,scales,view,k,width,height):
        ctx.save_for_backward(means,quats,scales,view,k);ctx.size=(width,height)
        activation_counts['forward']+=1
        return extension(True,True).forward(means,quats,scales,view,k,width,height)

    @staticmethod
    @once_differentiable
    def backward(ctx,gradient):
        result=extension(True,True).backward(*ctx.saved_tensors,gradient.contiguous(),*ctx.size)
        activation_counts['backward']+=1
        activation_counts['coordinate_backward']=activation_counts.get('coordinate_backward',0)+1
        return *result,None,None,None,None


def fields(means,quats,scales,view,k,*,width=777,height=581,internal_fp64=False,coordinate_pullback=False):
    values=(means,quats,scales,view,k)
    if any(v.device.type!='cuda' or v.dtype!=torch.float32 for v in values):
        raise ValueError('Native geometry requires CUDA FP32; no fallback')
    if view.requires_grad or k.requires_grad:
        raise ValueError('This method has fixed camera calibration')
    if coordinate_pullback and not internal_fp64:raise ValueError('Coordinate pullback requires approved internal FP64')
    implementation=_Fields64Coordinates if coordinate_pullback else (_Fields64 if internal_fp64 else _Fields)
    return implementation.apply(*(v.contiguous() for v in values),width,height)
