"""Gaussian-window SSIM for a standalone trainer. SPDX-License-Identifier: Apache-2.0"""
import torch
import torch.nn.functional as F


def foreground_photometric_inputs(rendered, target, foreground):
    """Match the standalone black, foreground-only RGB loss convention."""
    if rendered.ndim!=3 or rendered.shape[-1]!=3 or target.shape!=rendered.shape:
        raise ValueError('Photometric inputs must share HWC RGB shape')
    if foreground.shape!=rendered.shape[:2]:
        raise ValueError('Foreground and RGB pixel coordinates must match')
    if foreground.dtype!=torch.bool:
        raise TypeError('Foreground support must be boolean, not inferred from RGB')
    if rendered.dtype!=target.dtype or not rendered.is_floating_point():
        raise TypeError('Photometric RGB inputs must share floating dtype')
    if rendered.device!=target.device or foreground.device!=rendered.device:
        raise ValueError('Photometric inputs and foreground must share a device')
    support=foreground[...,None]
    return rendered*support,target*support


def structural_similarity_target_statistics(b):
    """Unchanged window statistics for an immutable, non-learned RGB target."""
    if b.requires_grad:
        raise ValueError('Cached SSIM target must not require gradients')
    b = b.permute(2, 0, 1)[None]
    coordinate = torch.arange(11, device=b.device, dtype=b.dtype)-5
    weights = torch.exp(-coordinate.square()/(2*1.5**2));weights = weights/weights.sum()
    kernel = (weights[:, None]*weights[None, :]).expand(3, 1, 11, 11).contiguous()
    mean = F.conv2d(b, kernel, padding=5, groups=3)
    variance = F.conv2d(b*b, kernel, padding=5, groups=3)-mean*mean
    return mean, variance


def structural_similarity(a, b, target_mean=None, target_variance=None, *, separable=False):
    a, b = a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None]
    coordinate = torch.arange(11, device=a.device, dtype=a.dtype)-5
    weights = torch.exp(-coordinate.square()/(2*1.5**2));weights = weights/weights.sum()
    if separable:
        horizontal = weights.reshape(1, 1, 1, 11).expand(3, 1, 1, 11).contiguous()
        vertical = weights.reshape(1, 1, 11, 1).expand(3, 1, 11, 1).contiguous()
        def average(value):
            value = F.conv2d(value, horizontal, padding=(0, 5), groups=3)
            return F.conv2d(value, vertical, padding=(5, 0), groups=3)
    else:
        kernel = (weights[:, None]*weights[None, :]).expand(3, 1, 11, 11).contiguous()
        def average(value):return F.conv2d(value, kernel, padding=5, groups=3)
    ma = average(a)
    mb = average(b) if target_mean is None else target_mean
    va = average(a*a)-ma*ma
    vb = average(b*b)-mb*mb if target_variance is None else target_variance
    covariance = average(a*b)-ma*mb
    value = ((2*ma*mb+.01**2)*(2*covariance+.03**2))/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2)).clamp_min(1e-12)
    return value.mean()
