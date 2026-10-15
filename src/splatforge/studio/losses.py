"""Reusable supervision modules independent of trainer and method registry."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_mean(value, mask):
    weight = mask.to(value.dtype)
    selected = torch.where(weight != 0, value, 0)
    return (selected * weight).sum() / weight.expand_as(value).sum().clamp_min(1)


def ssim_loss(prediction, target):
    # A contiguous HWC render permuted to NCHW is channels-last, not ordinary
    # NCHW contiguous. The current Torch 2.1 CUDA pooling backward diverges from
    # actual-image directional finite differences in that path. Make the loss
    # layout explicit without changing dtype, values, kernel or padding.
    a = prediction.permute(2, 0, 1).contiguous()[None]
    b = target.permute(2, 0, 1).contiguous()[None]
    kernel = min(11, a.shape[-2], a.shape[-1])
    kernel -= 1 - kernel % 2

    def pool(value):
        return F.avg_pool2d(value, kernel, stride=1, padding=kernel // 2)

    ma, mb = pool(a), pool(b)
    va, vb = pool(a * a) - ma * ma, pool(b * b) - mb * mb
    covariance = pool(a * b) - ma * mb
    score = (
        (2 * ma * mb + 0.01**2)
        * (2 * covariance + 0.03**2)
        / ((ma * ma + mb * mb + 0.01**2) * (va + vb + 0.03**2)).clamp_min(1e-8)
    )
    return (1 - score).mean()


def photometric_loss(prediction, target, ssim_weight):
    return (1 - ssim_weight) * (
        prediction - target
    ).abs().mean() + ssim_weight * ssim_loss(prediction, target)


def normal_consistency(output, camera, ops, mask):
    normal, valid = ops("geometry.depth_normals@1", output.depth.squeeze(-1), camera.K)
    valid = valid & (output.alpha.squeeze(-1) > 0.1) & (mask.squeeze(-1) > 0)
    if output.normal_numerator is not None:
        # Sum_i w_i (1 - n_i dot n_depth), not normalization of sum_i w_i n_i.
        return masked_mean(
            output.alpha.squeeze(-1) - (output.normal_numerator * normal).sum(-1), valid
        )
    return masked_mean(1 - (output.normals * normal).sum(-1), valid)


def patch_depth_loss(prediction, prior, valid, patch_size=32):
    loss = prediction.sum() * 0
    count = 0
    for top in range(0, prediction.shape[0], patch_size):
        for left in range(0, prediction.shape[1], patch_size):
            p = prediction[top : top + patch_size, left : left + patch_size]
            q = prior[top : top + patch_size, left : left + patch_size]
            m = valid[top : top + patch_size, left : left + patch_size].to(p.dtype)
            n = m.sum().clamp_min(1)
            mp = (p * m).sum() / n
            mq = (q * m).sum() / n
            centered_p = p - mp
            centered_q = q - mq
            scale = (centered_p * centered_q * m).sum() / (
                centered_p.square() * m
            ).sum().clamp_min(1e-8)
            offset = mq - scale * mp
            loss += masked_mean((scale * p + offset - q).abs(), m)
            count += 1
    return loss / max(count, 1)
