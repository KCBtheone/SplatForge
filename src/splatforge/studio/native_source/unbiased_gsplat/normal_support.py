"""Owned optional support for depth normals across occlusion discontinuities.

SPDX-License-Identifier: Apache-2.0
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from torch import Tensor


def continuous_normal_stencil(depth: Tensor, valid: Tensor, *, max_depth_jump: float=.008) -> Tensor:
    """Certify the original five-point stencil without bridging camera-z jumps.

    This is a nondifferentiable eligibility decision, not a new normal estimator.
    The fixed .008 in the Object1 experiment is its existing normalized TSDF
    truncation. It is not inferred from labels, a reference mesh or GT.
    """
    if depth.ndim<2 or depth.shape!=valid.shape:
        raise ValueError('depth and valid must have identical [...,H,W] shape')
    if not depth.dtype.is_floating_point or valid.dtype!=torch.bool:
        raise TypeError('depth must be floating and valid boolean')
    if depth.device!=valid.device:
        raise ValueError('depth and valid must share a device')
    if not math.isfinite(max_depth_jump) or max_depth_jump<=0:
        raise ValueError('max_depth_jump must be finite and positive')
    if min(depth.shape[-2:])<3:return torch.zeros_like(valid)
    with torch.no_grad():
        d=depth.detach();v=valid&torch.isfinite(d)&(d>0)
        center=d[...,1:-1,1:-1]
        supported=v[...,1:-1,1:-1].clone()
        for neighbor,nv in ((d[...,1:-1,:-2],v[...,1:-1,:-2]),
                            (d[...,1:-1,2:],v[...,1:-1,2:]),
                            (d[...,:-2,1:-1],v[...,:-2,1:-1]),
                            (d[...,2:,1:-1],v[...,2:,1:-1])):
            supported &= nv&((center-neighbor).abs()<=max_depth_jump)
        return F.pad(supported,(1,1,1,1),value=False)
