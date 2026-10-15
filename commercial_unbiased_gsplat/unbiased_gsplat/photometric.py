"""Gaussian-window SSIM for a standalone trainer. SPDX-License-Identifier: Apache-2.0"""
import torch
import torch.nn.functional as F


def structural_similarity(a, b):
    a, b = a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None]
    coordinate = torch.arange(11, device=a.device, dtype=a.dtype)-5
    weights = torch.exp(-coordinate.square()/(2*1.5**2));weights = weights/weights.sum()
    kernel = (weights[:, None]*weights[None, :]).expand(3, 1, 11, 11).contiguous()
    def average(value):return F.conv2d(value, kernel, padding=5, groups=3)
    ma, mb = average(a), average(b)
    va, vb = average(a*a)-ma*ma, average(b*b)-mb*mb
    covariance = average(a*b)-ma*mb
    value = ((2*ma*mb+.01**2)*(2*covariance+.03**2))/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2)).clamp_min(1e-12)
    return value.mean()
