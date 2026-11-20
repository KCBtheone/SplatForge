"""Independent multi-view supervision of each native surface, without mixing.

SPDX-License-Identifier: Apache-2.0
This is an engineering extension of our paper-derived multi-view operator,
not a claim about the Unbiased authors' training implementation.
"""
from __future__ import annotations
from .multiview import PGSRMultiViewConfig, multiview_consistency, weighted_multiview_regularization
from .regional_commercial import native_visibility


def active_multiview_families(layers: dict, policy: str = 'pgsr') -> tuple[str, ...]:
    if policy not in ['pgsr', 'both']:
        raise ValueError('Multi-view family policy must be pgsr or both')
    if not layers or set(layers) - {'pgsr', 'unbiased'}:
        raise ValueError('Multi-view requires known native families')
    requested = ('pgsr',) if policy == 'pgsr' else ('pgsr', 'unbiased')
    return tuple(f for f in requested if f in layers)


def native_multiview_constraints(reference_layers: dict, neighbor_layers: dict,
                                reference_item: dict, neighbor_item: dict, *,
                                policy: str = 'pgsr',
                                config: PGSRMultiViewConfig = PGSRMultiViewConfig(),
                                joint_physical_visibility: bool = False, hair_family: str = 'pgsr') -> tuple:
    """Match a family only to itself, using physical foreground in both views.

    The reference native geometry retains gradients. Neighbor geometry is
    explicitly detached, matching the existing one-sided training protocol.
    Semantic labels and the combined RGB/depth/alpha never certify geometry.
    Each active family receives the same existing 0.03 geometry / 0.15 photo
    coefficients; adding a family does not rescale the PGSR constraint.
    """
    from .dual_geometry import geometry_layers
    reference_layers,neighbor_layers=geometry_layers(reference_layers),geometry_layers(neighbor_layers)
    active = active_multiview_families(reference_layers, policy)
    if any(f not in neighbor_layers for f in active):
        raise ValueError('Neighbor must supply every active native family')
    if joint_physical_visibility and len(reference_layers)==2 and set(neighbor_layers)!=set(reference_layers):
        raise ValueError('Joint physical visibility requires both native neighbor layers')
    reference_visibility=native_visibility(reference_layers,hair_family) if joint_physical_visibility else None
    neighbor_visibility=native_visibility(neighbor_layers,hair_family) if joint_physical_visibility else None
    results = {}
    terms = []
    for family in active:
        source, target = reference_layers[family], neighbor_layers[family]
        result = multiview_consistency(reference_item['rgb'], neighbor_item['rgb'],
            source['points'], source['normals'], source['valid'],
            target['points'].detach(), target['normals'].detach(), target['valid'],
            reference_item['foreground'] if reference_visibility is None else reference_item['foreground']&reference_visibility[family],
            neighbor_item['foreground'] if neighbor_visibility is None else neighbor_item['foreground']&neighbor_visibility[family],
            reference_item['c2w'], neighbor_item['c2w'], reference_item['K'], neighbor_item['K'], config=config,
            reference_world_to_camera=reference_item.get('world_to_camera'),
            neighbor_world_to_camera=neighbor_item.get('world_to_camera'),
            reference_intrinsics_inverse=reference_item.get('intrinsics_inverse'),
            neighbor_intrinsics_inverse=neighbor_item.get('intrinsics_inverse'))
        results[family] = result
        terms.append(weighted_multiview_regularization(result, config))
    if not terms:
        return next(iter(reference_layers.values()))['points'].sum() * 0., results
    loss = terms[0]
    for term in terms[1:]:
        loss = loss + term
    return loss, results
