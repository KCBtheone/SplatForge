"""Optional initial tangent directions from ordinary input SfM neighborhoods.

SPDX-License-Identifier: Apache-2.0
No point, scale, opacity, colour, learned surface or evaluator is changed.
"""
from __future__ import annotations
import math
import numpy as np
import torch


@torch.no_grad()
def sfm_plane_quaternions(points, fallback, *, neighbors=16, minimum_eigengap=.1):
    """Keep original random orientations wherever a local plane is unresolved."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or fallback.shape != (len(points), 4):
        raise ValueError('SfM orientation needs points[N,3] and fallback wxyz[N,4]')
    if (not np.isfinite(points).all() or not fallback.is_floating_point()
            or not bool(torch.isfinite(fallback).all())
            or bool((fallback.square().sum(-1) <= 0).any())):
        raise ValueError('SfM orientation inputs must be finite floating values')
    if type(neighbors) is not int or neighbors < 2:
        raise ValueError('A tangent patch needs at least two neighboring points')
    if not math.isfinite(minimum_eigengap) or not 0 < minimum_eigengap <= 1:
        raise ValueError('Invalid relative plane eigengap')
    result = fallback.clone()
    reliable = np.zeros(len(points), dtype=bool)
    if len(points) >= 3:
        from scipy.spatial import cKDTree
        _, indices = cKDTree(points).query(points, k=min(neighbors+1, len(points)))
        patches = points[indices[:, 1:]]
        centered = patches-patches.mean(axis=1, keepdims=True)
        covariance = np.einsum('nki,nkj->nij', centered, centered)/patches.shape[1]
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        gap = np.divide(eigenvalues[:, 1]-eigenvalues[:, 0], eigenvalues[:, 2],
                        out=np.zeros(len(points)), where=eigenvalues[:, 2] > 0)
        normals = eigenvectors[:, :, 0]
        reliable = np.isfinite(normals).all(axis=1) & (eigenvalues[:, 2] > 0) & (gap >= minimum_eigengap)
        normals = normals[reliable]
        # Plane sign is immaterial. This hemisphere keeps the z-to-normal
        # quaternion away from its antipodal singularity without a camera prior.
        largest = np.argmax(np.abs(normals), axis=1)
        normals *= np.where(normals[np.arange(len(normals)), largest] < 0, -1., 1.)[:, None]
        w = np.sqrt(np.maximum(0., (1+normals[:, 2])/2))
        quaternions = np.column_stack((w, -normals[:, 1]/(2*w), normals[:, 0]/(2*w), np.zeros(len(w))))
        result[torch.as_tensor(reliable, device=fallback.device)] = torch.as_tensor(
            quaternions, device=fallback.device, dtype=fallback.dtype)
    return result, dict(method='sfm_plane', points=len(points), neighbors=neighbors,
        minimum_eigengap=minimum_eigengap, oriented_points=int(reliable.sum()),
        random_fallback_points=int((~reliable).sum()), points_removed=0,
        source='Original input SfM neighborhoods only; no learned geometry or GT')
