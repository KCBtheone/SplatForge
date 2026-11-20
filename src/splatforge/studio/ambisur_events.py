"""AmbiSuR event evaluation with the original projected-center adjoint.

The support bounds remove empty member slots only. The pixel axis, center
broadcast and final truncation/depth predicates retain the reference layout.
"""

from collections import defaultdict

import torch


class CenterLayoutEvents:
    def __init__(self):
        self.counts = defaultdict(int)

    def __call__(self, *args):
        self.counts["producer_calls"] += 1
        return center_layout_events(*args, counts=self.counts)


def center_layout_events(
    ids, valid_slots, tile_rays, pixels, centers, conics, opacities,
    normals, plane_distance, extent, near, far, *, counts,
):
    """Evaluate the verified center-layout component without a diagnostic hook."""
    if centers.dtype != torch.float32 or valid_slots.dtype != torch.bool:
        raise ValueError("original float32/valid slot inputs required")
    batches, members = ids.shape
    pixel_count = pixels.shape[1]
    with torch.no_grad():
        xy = centers[ids]
        q = conics[ids]
        a = q[..., 0, 0]
        b = (q[..., 0, 1] + q[..., 1, 0]) * .5
        c = q[..., 1, 1]
        det = a * c - b.square()
        safe = ((a > 0) & (c > 0) & (det > 0)
                & torch.isfinite(q).all(-1).all(-1)
                & torch.isfinite(xy).all(-1))
        denominator = torch.where(safe, det, torch.ones_like(det))
        rx = extent * (torch.where(safe, c, torch.ones_like(c)) / denominator).sqrt()
        ry = extent * (torch.where(safe, a, torch.ones_like(a)) / denominator).sqrt()
        safe = safe & torch.isfinite(rx) & torch.isfinite(ry)
        low = pixels.amin(1)
        high = pixels.amax(1)
        overlap = ((xy[..., 0] + rx + 1. >= low[:, 0, None])
                   & (xy[..., 0] - rx - 1. <= high[:, 0, None])
                   & (xy[..., 1] + ry + 1. >= low[:, 1, None])
                   & (xy[..., 1] - ry - 1. <= high[:, 1, None]))
        keep = valid_slots & (~safe | overlap)
        compact_count = int(keep.sum(1).max())
        rows, slots = keep.nonzero(as_tuple=True)
        rank = keep.long().cumsum(1) - 1
        columns = rank[rows, slots]
        compact_ids = torch.zeros((batches, compact_count), dtype=ids.dtype, device=ids.device)
        compact_valid = torch.zeros((batches, compact_count), dtype=torch.bool, device=ids.device)
        compact_ids[rows, columns] = ids[rows, slots]
        compact_valid[rows, columns] = True
        counts["slot_original_lanes"] += batches * members
        counts["slot_compact_lanes"] += batches * compact_count
        counts["slot_original_valid"] += int(valid_slots.sum())
        counts["slot_retained_valid"] += int(keep.sum())

    pixel_normals = normals[ids][:, :, None, :].expand(-1, -1, pixel_count, -1)
    if compact_count == 0:
        counts["slot_empty_batches"] += 1
        zero = pixels.new_zeros((batches, members, pixel_count)) + centers.sum() * 0
        return zero, zero, pixel_normals

    # Keep the original B/N/P subtraction and its reduction into each center.
    original_delta = pixels[:, None] - centers[ids][:, :, None]
    delta = original_delta.new_zeros((batches, compact_count, pixel_count, 2)).index_put(
        (rows, columns), original_delta[rows, slots]
    )
    squared = torch.einsum("tlpi,tlij,tlpj->tlp", delta, conics[compact_ids], delta)
    response = torch.exp(-.5 * squared)
    normal = normals[compact_ids][:, :, None, :].expand(-1, -1, pixel_count, -1)
    denominator = (normal * tile_rays[:, None]).sum(-1)
    safe = torch.where(denominator.abs() > 1e-8, denominator, torch.ones_like(denominator))
    depth = plane_distance[compact_ids][:, :, None] / safe
    response = torch.where(denominator.abs() > 1e-8, response, 0)
    response = torch.where(
        compact_valid[:, :, None] & (squared <= extent ** 2)
        & (depth > near) & (depth < far), response, 0
    )
    alpha = opacities[compact_ids][:, :, None] * response
    # Scatter real slots only; padding must not overwrite the first member.
    restored_alpha = alpha.new_zeros((batches, members, pixel_count)).index_put(
        (rows, slots), alpha[rows, columns]
    )
    restored_depth = depth.new_zeros((batches, members, pixel_count)).index_put(
        (rows, slots), depth[rows, columns]
    )
    return restored_alpha, restored_depth, pixel_normals
