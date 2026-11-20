"""Independent commercial regional composition and loss operators.

SPDX-License-Identifier: Apache-2.0
This explicitly retains two aggregate layers, not cross-primitive sorting.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import Tensor
from .background import composite_background
from .family_roles import native_family_roles


def geometry_domain(foreground: Tensor, unknown: Tensor, *, hair_confidence: Tensor | None = None) -> Tensor:
    """Semantic hair abstention restricts only the regional hair model."""
    if hair_confidence is None:
        return foreground
    return foreground & ~unknown & (hair_confidence > 0)


def physical_geometry_domain(foreground: Tensor) -> Tensor:
    """Object existence is observed even when family identity is unlabelled.

    Native depth/normal validity is applied by the geometric loss. Semantic
    identity confidence must not erase this observed physical foreground.
    """
    return foreground


def appearance_gradient(value: Tensor, weight: Tensor) -> Tensor:
    """Identity forward for a scalar field or last-axis appearance channels."""
    if value.ndim == weight.ndim + 1:
        weight = weight[..., None]
    elif value.shape != weight.shape:
        raise ValueError('Appearance weights must match the scalar field or channel support')
    return value.detach() + weight*(value-value.detach())


def native_visibility(layers: dict, hair_family: str = 'pgsr', *, selected_surfel: Tensor | None = None) -> dict:
    """Keep a native surface observable unless the other opaque layer is in front."""
    if len(layers)==1:
        return {key:torch.ones_like(layer['valid']) for key,layer in layers.items()}
    if set(layers)!={'pgsr','unbiased'}:
        raise ValueError('Joint native visibility requires both known families')
    hf,bf=native_family_roles(hair_family);hair,body=layers[hf],layers[bf]
    hair_front=hair['valid']&(~body['valid']|(hair['depth']<=body['depth']))
    visibility={hf:~(body['valid']&(body['alpha']>=.5)&~hair_front),
                bf:~(hair['valid']&(hair['alpha']>=.5)&hair_front)}
    if selected_surfel is not None:
        surfel=layers['unbiased']
        if selected_surfel.dtype!=torch.bool or selected_surfel.shape!=surfel['valid'].shape or selected_surfel.device!=surfel['valid'].device:
            raise ValueError('Selected Surfel support must match its native validity mask')
        # Photometric forward already selects this native surface. An excluded
        # family's raw layer must not erase the selected surface's geometry.
        visibility['unbiased']=visibility['unbiased'] | (selected_surfel & surfel['valid'])
    return visibility


def normal_stencil_visibility(visible: Tensor) -> Tensor:
    """The observed central-difference normal needs all five surface samples."""
    stencil=torch.zeros_like(visible)
    stencil[1:-1,1:-1]=(visible[1:-1,1:-1]&visible[:-2,1:-1]&visible[2:,1:-1]
                        &visible[1:-1,:-2]&visible[1:-1,2:])
    return stencil


def visible_native_geometry_losses(terms: dict, visible: Tensor) -> tuple:
    """Preserve native scalar values/reductions, restrict unsupported pixel VJPs."""
    convergence=appearance_gradient(terms['convergence'],visible.flatten()).sum()/terms['convergence_denominator'].clamp_min(1)
    normal=appearance_gradient(terms['normal'],normal_stencil_visibility(visible).flatten()).sum()/terms['normal_denominator'].clamp_min(1)
    return convergence,normal


def regional_identity_policy(foreground: Tensor, hair_confidence: Tensor, known: Tensor,
                             unknown: Tensor, *, trusted_hair: Tensor, trusted_body: Tensor) -> dict:
    """Abstain from family identity where an observed foreground has no label.

    Unlabelled physical RGB keeps equal complete RGBA derivatives. Independent
    per-family RGB anchors use only supported identity or observed background.
    These weights never modify forward RGB, native geometry, or semantic terms.
    """
    if any(value.shape != foreground.shape for value in [hair_confidence, known, unknown, trusted_hair, trusted_body]):
        raise ValueError('Regional identity fields must have the same pixel support')
    if any(value.dtype != torch.bool for value in [foreground, known, unknown, trusted_hair, trusted_body]):
        raise ValueError('Regional identity domains must be boolean')
    supported = foreground & known & ~unknown
    positive = hair_confidence.detach().clamp(0, 1)
    unit = torch.ones_like(positive)
    pg_weight = torch.where(supported, positive.clamp_min(.1), unit)
    ug_weight = torch.where(supported, (1-positive).clamp_min(.1), unit)
    anchor_support = (~foreground) | supported
    pg_anchor = pg_weight * anchor_support * (~(trusted_body & supported))
    ug_anchor = ug_weight * anchor_support * (~(trusted_hair & supported))
    return dict(appearance_weights=(pg_weight, ug_weight),
                native_anchor_weights=(pg_anchor, ug_anchor), identity_supported=supported)


def native_radiance_anchor(layer: dict, target: Tensor, weight: Tensor, *, background: Tensor,
                           exposure: Tensor | None = None) -> Tensor:
    """Observe each family's own radiance even when the other layer occludes it.

    Fixed appearance-confidence weights, including background, are supplied by
    the regional policy. An opposite trusted identity has zero weight. Neither
    current opacity nor depth validity gates this loss: missing opacity must
    retain its gradient. Premultiplied RGB and alpha share the physical path.
    """
    if target.shape != layer['rgb'].shape or weight.shape != layer['alpha'].shape:
        raise ValueError('Native anchor shapes must match RGB and alpha support')
    if target.shape[:-1] != weight.shape:
        raise ValueError('Native anchor RGB must have one last channel axis')
    native = composite_background(layer['rgb'], layer['alpha'], background)
    if exposure is not None:
        native = exposure[0].exp()*native + exposure[1]
    difference = (native-target).abs().mean(-1)
    return (difference*weight).sum()/weight.sum().clamp_min(1)


def select_opaque_geometry(hair: dict, body: dict, *, minimum_alpha: float = .5) -> dict:
    """Choose the nearest valid native surface supported by its own opacity.

    This post-render geometry decision is independent of trusted 2D identity
    guards. A different family's alpha cannot certify an opaque surface here.
    RGB composition, ownership probabilities and their gradients are untouched.
    """
    if not 0 < minimum_alpha <= 1:
        raise ValueError('Geometry opacity threshold must lie in (0,1]')
    def candidate(layer):
        return (layer['valid'] & torch.isfinite(layer['depth']) & (layer['depth'] > 0)
                & torch.isfinite(layer['alpha']) & (layer['alpha'] >= minimum_alpha))
    vh, vb = candidate(hair), candidate(body)
    selected_hair = vh & (~vb | (hair['depth'] <= body['depth']))
    valid = vh | vb
    depth = torch.where(selected_hair, hair['depth'], body['depth'])
    alpha = torch.where(selected_hair, hair['alpha'], body['alpha'])
    return dict(depth=torch.where(valid, depth, torch.zeros_like(depth)),
                alpha=torch.where(valid, alpha, torch.zeros_like(alpha)),
                valid=valid, hair_selected=selected_hair)


def compose_two_families(hair: dict, body: dict, *, trusted_hair: Tensor | None = None,
                         trusted_body: Tensor | None = None, background: Tensor | None = None,
                         appearance_weights: tuple[Tensor, Tensor] | None = None,
                         occlusion_aware_identity: bool = False,
                         _constructed_disjoint_domains: bool = False) -> dict:
    ah, ab = hair['alpha'], body['alpha']
    vh, vb = hair['valid'], body['valid']
    hair_front = vh & (~vb | (hair['depth'] <= body['depth']))
    wh = torch.where(hair_front, ah, ah*(1-ab))
    wb = torch.where(hair_front, ab*(1-ah), ab)
    probability = wh / (wh+wb).clamp_min(1e-8)
    # Weight the complete photometric RGBA derivative, including transmission.
    # Geometry and native semantic/leakage opacity remain unweighted.
    if appearance_weights is None:
        ch, cb, aph, apb = hair['rgb'], body['rgb'], ah, ab
    else:
        weight_hair, weight_body = appearance_weights
        ch = appearance_gradient(hair['rgb'], weight_hair)
        cb = appearance_gradient(body['rgb'], weight_body)
        aph = appearance_gradient(ah, weight_hair)
        apb = appearance_gradient(ab, weight_body)
    native_rgb = torch.where(hair_front[..., None],
                             ch+(1-aph[..., None])*cb,
                             cb+(1-apb[..., None])*ch)
    native_depth = torch.where(hair_front, hair['depth'], body['depth'])
    native_alpha = ah + ab - ah*ab
    photo_native_alpha = aph + apb - aph*apb
    if trusted_hair is None:
        trusted_hair = torch.zeros_like(vh)
    if trusted_body is None:
        trusted_body = torch.zeros_like(vb)
    if not _constructed_disjoint_domains and bool((trusted_hair & trusted_body).any()):
        raise ValueError('Trusted hair and body domains overlap')
    if occlusion_aware_identity:
        # Identity labels describe the visible surface, not hidden 3D material.
        # Keep their forward penalties and the front family's derivative, while
        # preventing a back view of hair from erasing the face behind it.
        hidden_hair = trusted_body & vb & ~hair_front
        hidden_body = trusted_hair & vh & hair_front
        identity_ah = appearance_gradient(ah, (~hidden_hair).to(ah.dtype))
        identity_ab = appearance_gradient(ab, (~hidden_body).to(ab.dtype))
        wh = torch.where(hair_front, identity_ah, identity_ah*(1-identity_ab))
        wb = torch.where(hair_front, identity_ab*(1-identity_ah), identity_ab)
        probability = wh / (wh+wb).clamp_min(1e-8)
    guarded_rgb = torch.where(trusted_hair[..., None], ch,
                              torch.where(trusted_body[..., None], cb, native_rgb))
    selected_hair = torch.where(trusted_hair, torch.ones_like(hair_front),
                                torch.where(trusted_body, torch.zeros_like(hair_front), hair_front))
    guarded_alpha = torch.where(trusted_hair, ah, torch.where(trusted_body, ab, native_alpha))
    photo_guarded_alpha = torch.where(trusted_hair, aph, torch.where(trusted_body, apb, photo_native_alpha))
    return dict(rgb=composite_background(guarded_rgb, photo_guarded_alpha, background),
                native_rgb=composite_background(native_rgb, photo_native_alpha, background), alpha=guarded_alpha,
                depth=torch.where(selected_hair, hair['depth'], body['depth']),
                valid=torch.where(selected_hair, vh, vb),
                probability=probability, hair_selected=selected_hair,
                native_depth=native_depth, native_alpha=native_alpha,
                visible_hair_alpha=wh, visible_body_alpha=wb,
                occlusion_aware_identity=occlusion_aware_identity)


def semantic_and_leakage(composite: dict, trusted_hair: Tensor, trusted_body: Tensor, *,
                         leakage_reduction: str = 'violators') -> tuple[Tensor, Tensor]:
    """Average excess over fixed trusted support, or retain the legacy mean."""
    if leakage_reduction not in ['violators', 'trusted-support']:
        raise ValueError('Unknown leakage reduction policy')
    p = composite['probability'].clamp(1e-6, 1-1e-6)
    support = trusted_hair | trusted_body
    semantic = F.binary_cross_entropy(p, trusted_hair.to(p.dtype), reduction='none')
    semantic = (semantic * support).sum() / support.sum().clamp_min(1)
    violation = torch.where(trusted_body, (composite['visible_hair_alpha']-.05).clamp_min(0),
                            torch.where(trusted_hair, (composite['visible_body_alpha']-.05).clamp_min(0), torch.zeros_like(p)))
    denominator = support.sum() if leakage_reduction == 'trusted-support' else (violation.detach() > 0).sum()
    leakage = violation.sum() / denominator.clamp_min(1)
    return semantic, leakage


def photometric_loss(rendered: Tensor, target: Tensor, exposure: Tensor | None = None, *,
                     structural_similarity_function=None, return_exposure: bool = False):
    from .photometric import structural_similarity
    similarity = structural_similarity if structural_similarity_function is None else structural_similarity_function
    structural = 1-similarity(rendered, target)
    adjusted = rendered
    effective_exposure = None
    if exposure is not None:
        effective_exposure = torch.where(structural.detach() < .5, exposure, torch.zeros_like(exposure))
        adjusted = effective_exposure[0].exp()*rendered + effective_exposure[1]
    loss = .8*(adjusted-target).abs().mean() + .2*structural
    return (loss, effective_exposure) if return_exposure else loss
