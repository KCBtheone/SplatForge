"""Torch GOF event evaluation with a first-order float32 CUDA VJP."""
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

from ...._paths import output_path

_extension = None
activation_counts = {'ordered_forward': 0, 'ordered_backward': 0}


def extension():
    global _extension
    if _extension is None:
        from torch.utils.cpp_extension import load
        root=os.environ.get('TORCH_EXTENSIONS_DIR')
        if not root:
            raise RuntimeError('Set TORCH_EXTENSIONS_DIR before loading the GOF CUDA extension')
        build = output_path(Path(root) / 'gof_ordered_h005')
        build.mkdir(parents=True,exist_ok=True)
        _extension=load(name='sf_gof_ordered_h005',sources=[str(Path(__file__).with_suffix('.cu'))],
                        build_directory=str(build),extra_cflags=['-O3'],
                        extra_cuda_cflags=['-O3','--fmad=false'],verbose=False)
    return _extension


class OrderedEvents(torch.autograd.Function):
    @staticmethod
    def forward(ctx,proxy,qr,qe,opacity,rays,squared,valid,near,far,extent):
        activation_counts['ordered_forward']+=1
        qray=torch.einsum('tlij,tpj->tlpi',qr,rays)
        rawden=(qray*rays[:,None]).sum(-1)
        den=rawden.clamp_min(1e-12)
        num=(proxy*qray).sum(-1)
        depth=num/den
        offset=depth[...,None]*rays[:,None]-proxy
        exponent=torch.einsum('tlpi,tlij,tlpj->tlp',offset,qe,offset)
        response=torch.exp(-.5*exponent.clamp_min(0))
        mask=valid[:,:,None] & (squared<=extent**2) & (depth>near) & (depth<far)
        normal=F.normalize(-qray,dim=-1)
        left=torch.einsum('tlpi,tlij->tlpj',offset,qe)
        ctx.save_for_backward(proxy,qr,qe,opacity,rays,qray,rawden,den,num,depth,
                              offset,exponent,response,mask,normal,qray.norm(dim=-1),left)
        return opacity[:,:,None]*torch.where(mask,response,0),depth,normal

    @staticmethod
    @once_differentiable
    def backward(ctx,ga,gd,gn):
        activation_counts['ordered_backward']+=1
        return (*extension().backward(list(ctx.saved_tensors),ga,gd,gn),None,None,None,None,None)


def ordered_events(proxy,qr,qe,opacity,rays,squared,valid,*,near,far,extent):
    if any(x.device.type!='cuda' or x.dtype!=torch.float32 or x.device!=proxy.device
           for x in (proxy,qr,qe,opacity,rays,squared)):
        raise ValueError('explicit same-device CUDA float32 operands required')
    b,l,p,d=proxy.shape
    if (d!=3 or qr.shape!=(b,l,3,3) or qe.shape!=qr.shape or opacity.shape!=(b,l)
            or rays.shape!=(b,p,3) or squared.shape!=(b,l,p) or valid.shape!=(b,l)
            or valid.dtype!=torch.bool or valid.device!=proxy.device):
        raise ValueError('invalid ordered event operands')
    return OrderedEvents.apply(proxy,qr,qe,opacity,rays,squared,valid,near,far,extent)
