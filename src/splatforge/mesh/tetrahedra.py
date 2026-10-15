"""Generic tetrahedral level-set extraction with shared-edge bisection.

The field callable takes world-space NumPy points and returns one finite scalar
per point. Topology is discrete; this exporter does not claim a differentiable
mesh topology operation or CUDA Graph support.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.spatial import Delaunay


EDGES = np.asarray(list(combinations(range(4), 2)), np.int64)


def triangulate(points, radii=None):
    points = np.asarray(points, np.float64)
    if len(points) < 4 or np.linalg.matrix_rank(points - points.mean(0)) < 3:
        raise ValueError("tetrahedral extraction needs >=4 noncoplanar points")
    cells = Delaunay(points).simplices.astype(np.int64)
    if radii is not None:
        edges = cells[:, EDGES]
        distance = np.linalg.norm(
            points[edges[..., 0]] - points[edges[..., 1]], axis=-1
        )
        limits = np.asarray(radii)[edges[..., 0]] + np.asarray(radii)[edges[..., 1]]
        cells = cells[(distance <= limits + 1e-10).all(-1)]
    return cells


def marching_tetrahedra(
    points, cells, values, field, *, level=0.0, binary_steps=8, inside_greater=True
):
    points = np.asarray(points, np.float64)
    cells = np.asarray(cells, np.int64)
    values = np.asarray(values, np.float64)
    if values.shape != (len(points),) or not np.isfinite(values).all():
        raise ValueError("field returned invalid vertex scalars")
    inside = values >= level if inside_greater else values <= level
    active = (inside[cells].any(-1)) & (~inside[cells].all(-1))
    cells = cells[active]
    if not len(cells):
        return np.empty((0, 3)), np.empty((0, 3), np.int64)
    edges = cells[:, EDGES]
    crossing = inside[edges[..., 0]] != inside[edges[..., 1]]
    pairs = np.sort(edges[crossing], axis=-1)
    unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
    low = points[unique[:, 0]].copy()
    high = points[unique[:, 1]].copy()
    low_inside = inside[unique[:, 0]]
    for _ in range(binary_steps):
        middle = (low + high) / 2
        evaluated = np.asarray(field(middle))
        if evaluated.shape != (len(middle),) or not np.isfinite(evaluated).all():
            raise ValueError("field returned invalid bisection scalars")
        middle_inside = evaluated >= level if inside_greater else evaluated <= level
        same = middle_inside == low_inside
        low[same] = middle[same]
        high[~same] = middle[~same]
    vertices = (low + high) / 2
    edge_indices = np.full((len(cells), 6), -1, np.int64)
    edge_indices[crossing] = inverse
    faces = []
    pair_index = {tuple(pair): index for index, pair in enumerate(EDGES.tolist())}

    def get(row, a, b):
        return int(row[pair_index[tuple(sorted((int(a), int(b))))]])

    for cell, row in zip(cells, edge_indices):
        inn = np.flatnonzero(inside[cell])
        out = np.flatnonzero(~inside[cell])
        if len(inn) in (1, 3):
            pivot = inn[0] if len(inn) == 1 else out[0]
            rest = out if len(inn) == 1 else inn
            triangles = [[get(row, pivot, other) for other in rest]]
        else:
            a, b = inn
            c, d = out
            ac, ad, bc, bd = (
                get(row, a, c),
                get(row, a, d),
                get(row, b, c),
                get(row, b, d),
            )
            triangles = [[ac, ad, bc], [ad, bd, bc]]
        outward = points[cell[out]].mean(0) - points[cell[inn]].mean(0)
        for face in triangles:
            triangle = vertices[face]
            if (
                np.dot(
                    np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0]),
                    outward,
                )
                < 0
            ):
                face[1], face[2] = face[2], face[1]
            faces.append(face)
    faces = np.asarray(faces, np.int64)
    area = np.linalg.norm(
        np.cross(
            vertices[faces[:, 1]] - vertices[faces[:, 0]],
            vertices[faces[:, 2]] - vertices[faces[:, 0]],
        ),
        axis=-1,
    )
    return vertices, faces[area > 1e-16]
