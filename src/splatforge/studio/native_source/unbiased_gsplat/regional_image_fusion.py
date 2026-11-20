"""One fixed-resolution execution graph for the existing joint image objective.

SPDX-License-Identifier: Apache-2.0
Native reconstruction, normal construction and multi-view losses stay outside.
"""
from .regional_commercial import (
    compose_two_families, native_radiance_anchor, native_visibility,
    photometric_loss, regional_identity_policy, semantic_and_leakage,
    visible_native_geometry_losses,
)


def joint_image_objective(pgsr, unbiased, item, background, exposure):
    """Preserve the current PGSR-hair, physical, exposure-space objective."""
    layers = dict(pgsr=pgsr, unbiased=unbiased)
    visibility = native_visibility(layers, 'pgsr')
    convergence, normal = visible_native_geometry_losses(
        unbiased['geometry_terms'], visibility['unbiased'])
    trusted_hair = ((item['hair'] >= .99) | (item['detail'] >= .9)) & ~item['unknown']
    # Construction itself excludes overlap, including contradictory SAM fields.
    trusted_body = (item['body'] >= .99) & ~item['unknown'] & ~trusted_hair
    policy = regional_identity_policy(item['foreground'], item['hair'], item['known'], item['unknown'],
                                     trusted_hair=trusted_hair, trusted_body=trusted_body)
    composite = compose_two_families(
        pgsr, unbiased, trusted_hair=trusted_hair, trusted_body=trusted_body,
        background=background, appearance_weights=policy['appearance_weights'],
        occlusion_aware_identity=True, _constructed_disjoint_domains=True)
    semantic, leakage = semantic_and_leakage(composite, trusted_hair, trusted_body,
                                            leakage_reduction='trusted-support')
    color, effective = photometric_loss(composite['rgb'], item['rgb'], exposure, return_exposure=True)
    anchors = dict(pgsr=policy['native_anchor_weights'][0], unbiased=policy['native_anchor_weights'][1])
    pg_anchor = native_radiance_anchor(pgsr, item['rgb'], anchors['pgsr'],
                                       background=background, exposure=effective)
    ug_anchor = native_radiance_anchor(unbiased, item['rgb'], anchors['unbiased'],
                                       background=background, exposure=effective)
    composite.update(semantic_loss=semantic, leakage_loss=leakage,
                     trusted_hair=trusted_hair, trusted_body=trusted_body,
                     native_anchor_weights=anchors, native_pgsr_anchor=pg_anchor,
                     native_unbiased_anchor=ug_anchor, _regional_color=color,
                     _regional_effective_exposure=effective)
    return composite, visibility, convergence, normal
