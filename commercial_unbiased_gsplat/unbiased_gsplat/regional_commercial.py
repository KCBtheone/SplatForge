"""Independent commercial regional composition and loss operators.

SPDX-License-Identifier: Apache-2.0
This explicitly retains two aggregate layers, not cross-primitive sorting.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import Tensor


def compose_two_families(hair: dict, body: dict, *, trusted_hair: Tensor | None = None,
                         trusted_body: Tensor | None = None) -> dict:
    ah, ab = hair['alpha'], body['alpha']
    vh, vb = hair['valid'], body['valid']
    hair_front = vh & (~vb | (hair['depth'] <= body['depth']))
    wh = torch.where(hair_front, ah, ah*(1-ab))
    wb = torch.where(hair_front, ab*(1-ah), ab)
    probability = wh / (wh+wb).clamp_min(1e-8)
    native_rgb = torch.where(hair_front[..., None],
                             hair['rgb']+(1-ah[..., None])*body['rgb'],
                             body['rgb']+(1-ab[..., None])*hair['rgb'])
    native_depth = torch.where(hair_front, hair['depth'], body['depth'])
    native_alpha = ah + ab - ah*ab
    if trusted_hair is None:
        trusted_hair = torch.zeros_like(vh)
    if trusted_body is None:
        trusted_body = torch.zeros_like(vb)
    if bool((trusted_hair & trusted_body).any()):
        raise ValueError('Trusted hair and body domains overlap')
    guarded_rgb = torch.where(trusted_hair[..., None], hair['rgb'],
                              torch.where(trusted_body[..., None], body['rgb'], native_rgb))
    selected_hair = torch.where(trusted_hair, torch.ones_like(hair_front),
                                torch.where(trusted_body, torch.zeros_like(hair_front), hair_front))
    return dict(rgb=guarded_rgb, native_rgb=native_rgb, alpha=torch.where(trusted_hair, ah, torch.where(trusted_body, ab, native_alpha)),
                depth=torch.where(selected_hair, hair['depth'], body['depth']),
                valid=torch.where(selected_hair, vh, vb),
                probability=probability, hair_selected=selected_hair,
                native_depth=native_depth, native_alpha=native_alpha,
                visible_hair_alpha=wh, visible_body_alpha=wb)


def semantic_and_leakage(composite: dict, trusted_hair: Tensor, trusted_body: Tensor) -> tuple[Tensor, Tensor]:
    p = composite['probability'].clamp(1e-6, 1-1e-6)
    support = trusted_hair | trusted_body
    semantic = F.binary_cross_entropy(p, trusted_hair.to(p.dtype), reduction='none')
    semantic = (semantic * support).sum() / support.sum().clamp_min(1)
    violation = torch.where(trusted_body, (composite['visible_hair_alpha']-.05).clamp_min(0),
                            torch.where(trusted_hair, (composite['visible_body_alpha']-.05).clamp_min(0), torch.zeros_like(p)))
    leakage = violation.sum() / ((violation.detach() > 0).sum().clamp_min(1))
    return semantic, leakage


def photometric_loss(rendered: Tensor, target: Tensor, exposure: Tensor | None = None) -> Tensor:
    from .photometric import structural_similarity
    structural = 1-structural_similarity(rendered, target)
    adjusted = rendered
    if exposure is not None:
        adjusted = torch.where((structural.detach() < .5), exposure[0].exp()*rendered+exposure[1], rendered)
    return .8*(adjusted-target).abs().mean() + .2*structural
