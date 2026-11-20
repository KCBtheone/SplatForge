"""Experimental opt-in input-only multi-view photo constraint.

SPDX-License-Identifier: Apache-2.0
Owned extension of our permitted paper-derived patch operators. No neighbor
learned depth, semantic ID, reference mesh, or author implementation is read.
"""
from __future__ import annotations
import torch
from .multiview import (
    PGSRMultiViewConfig, validate_multiview_config, plane_induced_homography,
    apply_homography, normalized_cross_correlation, _in_bounds, _grayscale,
    _sample_hwc, _unfold_fixed_patches,
)


def input_photometric_consensus(layer, reference, neighbors, *, config=PGSRMultiViewConfig()):
    """Median original 7x7 input NCC over >=3 valid existing camera neighbors.

    The current reference surface defines plane warps and receives derivatives.
    Each neighbor supplies only original image, foreground, and fixed camera.
    No learned-neighbor geometry can exclude an incorrect reference depth.
    Original stride/patch/interpolation/variance/foreground constants stay.
    Trusted reference body centers are the requested Surfel material; patches
    themselves retain all observed foreground. This is an additional proposed
    constraint, not a reproduction of an author's Unbiased training loss.
    """
    validate_multiview_config(config)
    image = reference['rgb'].detach()
    h, w, _ = image.shape
    radius = config.patch_size//2
    zero = layer['points'].sum()*0 + layer['normals'].sum()*0
    if h <= 2*radius or w <= 2*radius:
        return dict(loss=zero, denominator=zero.detach(), pixels=image.new_empty((0, 2)),
                    valid=torch.zeros(0, dtype=torch.bool, device=image.device),
                    valid_neighbors=torch.zeros(0, dtype=torch.long, device=image.device))
    ys = torch.arange(radius, h-radius, config.pixel_stride, device=image.device)
    xs = torch.arange(radius, w-radius, config.pixel_stride, device=image.device)
    if not len(ys) or not len(xs):
        return dict(loss=zero, denominator=zero.detach(), pixels=image.new_empty((0, 2)),
                    valid=torch.zeros(0, dtype=torch.bool, device=image.device),
                    valid_neighbors=torch.zeros(0, dtype=torch.long, device=image.device))
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    yy, xx = yy.flatten(), xx.flatten()
    pixels = torch.stack((xx.to(image.dtype)+.5, yy.to(image.dtype)+.5), -1)
    points = layer['points'][yy, xx]
    normals = layer['normals'][yy, xx]
    center_valid = (layer['valid'][yy, xx] & reference['foreground'][yy, xx]
                    & reference['known'][yy, xx] & ~reference['unknown'][yy, xx]
                    & (reference['body'][yy, xx] >= .99)
                    & torch.isfinite(points).all(-1) & torch.isfinite(normals).all(-1)
                    & (normals.norm(dim=-1) > config.homography_epsilon))
    patches = _unfold_fixed_patches(_grayscale(image), patch_size=config.patch_size,
                                   pixel_stride=config.pixel_stride, expected_rows=len(pixels))
    mask_patches = _unfold_fixed_patches(reference['foreground'].to(image.dtype)[..., None],
        patch_size=config.patch_size, pixel_stride=config.pixel_stride, expected_rows=len(pixels))
    center_valid = center_valid & (mask_patches >= config.sampled_mask_threshold).all(-1)
    oy, ox = torch.meshgrid(torch.arange(-radius, radius+1, device=image.device),
                           torch.arange(-radius, radius+1, device=image.device), indexing='ij')
    offsets = torch.stack((ox, oy), -1).reshape(-1, 2).to(image.dtype)
    patch_pixels = pixels[:, None]+offsets[None]
    costs, valids = [], []
    for neighbor in neighbors:
        if neighbor['rgb'].shape != image.shape:
            raise ValueError('Input consensus requires identical image shape')
        homography, plane_valid = plane_induced_homography(reference['c2w'], neighbor['c2w'],
            reference['K'], neighbor['K'], points, normals, epsilon=config.homography_epsilon,
            source_world_to_camera=reference.get('world_to_camera'),
            target_world_to_camera=neighbor.get('world_to_camera'),
            source_intrinsics_inverse=reference.get('intrinsics_inverse'))
        warped, projection_valid = apply_homography(homography, patch_pixels, epsilon=config.homography_epsilon)
        source = torch.cat((_grayscale(neighbor['rgb'].detach()),
                            neighbor['foreground'].to(image.dtype)[..., None]), -1)
        sampled = _sample_hwc(source, warped)
        ncc, variance_valid = normalized_cross_correlation(patches, sampled[..., 0],
                                                          variance_epsilon=config.variance_epsilon)
        target_view = neighbor.get('world_to_camera')
        if target_view is None:target_view = torch.linalg.inv(neighbor['c2w'])
        target_z = points@target_view[2, :3]+target_view[2, 3]
        valid = (center_valid & plane_valid & projection_valid.all(-1)
                 & _in_bounds(warped, width=w, height=h).all(-1)
                 & (sampled[..., 1] >= config.sampled_mask_threshold).all(-1)
                 & variance_valid & (target_z > 0))
        costs.append(1-ncc)
        valids.append(valid)
    if not costs:
        return dict(loss=zero, denominator=zero.detach(), pixels=pixels,
            valid=torch.zeros(len(pixels), dtype=torch.bool, device=image.device),
            valid_neighbors=torch.zeros(len(pixels), dtype=torch.long, device=image.device))
    costs, valids = torch.stack(costs), torch.stack(valids)
    counts = valids.sum(0)
    ordered = torch.sort(torch.where(valids, costs, torch.full_like(costs, float('inf'))), dim=0).values
    lo = ((counts-1).clamp_min(0)//2)[None]
    hi = (counts//2)[None]
    median = .5*(ordered.gather(0, lo)[0]+ordered.gather(0, hi)[0])
    valid = counts >= 3
    terms = torch.where(valid, median, torch.zeros_like(median))
    denominator = valid.sum().to(image.dtype)
    return dict(loss=terms.sum()/denominator.clamp_min(1)+zero, denominator=denominator,
                pixels=pixels, valid=valid, valid_neighbors=counts)
