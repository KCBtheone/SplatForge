"""Independent MILo pivot/mesh chain, from paper v2 Eq. (1) and (2).

Connectivity is discrete. Continuous vertex positions retain gradients to the
Gaussian means, rotations, scales, and independent tanh-normalized SDF values.
The explicitly named SciPy triangulator is a CPU reference, not a CGAL claim.
"""

from dataclasses import dataclass
from itertools import combinations, product

import torch

from ..providers.torch.gaussian.reference import rotation_matrix


_EDGES = tuple(combinations(range(4), 2))


def _face_cases():
    lookup = {edge: i for i, edge in enumerate(_EDGES)}
    cases = []
    for mask in range(16):
        inside = [i for i in range(4) if mask & (1 << i)]
        outside = [i for i in range(4) if i not in inside]
        def edge(a, b):
            return lookup[tuple(sorted((a, b)))]
        if len(inside) in (1, 3):
            pivot, other = (inside[0], outside) if len(inside) == 1 else (outside[0], inside)
            row = [edge(pivot, j) for j in other]
        elif len(inside) == 2:
            a, b = inside
            c, d = outside
            row = [edge(a, c), edge(a, d), edge(b, c),
                   edge(a, d), edge(b, d), edge(b, c)]
        else:
            row = []
        cases.append(row + [-1] * (6 - len(row)))
    return cases


_FACES = _face_cases()


@dataclass
class PivotMesh:
    vertices: torch.Tensor
    faces: torch.Tensor
    source_edges: torch.Tensor


def gaussian_pivots(means, quaternions, log_scales, *, pivot_ids=None):
    """Eq. (1), center then eight (+/-1)^3 box corners, in world coordinates."""
    if means.ndim != 2 or means.shape[1] != 3 or quaternions.shape != (len(means), 4) or log_scales.shape != means.shape:
        raise ValueError("expected N Gaussian means, wxyz quaternions and raw log-scales")
    if pivot_ids is not None:
        means, quaternions, log_scales = (x[pivot_ids] for x in (means, quaternions, log_scales))
    box = means.new_tensor([(0., 0., 0.), *product((-1., 1.), repeat=3)])
    local = log_scales.exp()[:, None, :] * box[None, :, :]
    rotated = torch.einsum("nij,nkj->nki", rotation_matrix(quaternions), local)
    return means[:, None, :] + rotated


def triangulate_pivots_scipy_cpu_reference(points):
    """Explicit initial/reference topology; no jitter, pruning, or fallback."""
    from scipy.spatial import Delaunay

    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
        raise ValueError("Delaunay requires at least four 3D sites")
    cells = Delaunay(points.detach().cpu().numpy()).simplices.copy()
    return torch.as_tensor(cells, dtype=torch.long, device=points.device)


def marching_tetrahedra(points, cells, signed_values):
    """Eq. (2), unique shared edges and outward-facing triangles.

    Negative values are inside. Zero is on the exterior side of a crossing;
    topology selection is piecewise constant, while intersection positions use
    the original differentiable values without an epsilon or detached VJP.
    """
    if points.ndim != 2 or points.shape[1] != 3 or signed_values.shape != (len(points),):
        raise ValueError("expected Vx3 sites and one signed value per site")
    if cells.ndim != 2 or cells.shape[1] != 4 or cells.dtype != torch.long:
        raise ValueError("expected int64 Tx4 tetrahedra")
    if points.device != cells.device or points.device != signed_values.device:
        raise ValueError("sites, cells and SDF must share a device")
    inside = signed_values < 0
    signs = inside[cells]
    active = signs.any(-1) & ~signs.all(-1)
    cells, signs = cells[active], signs[active]
    edges = cells[:, torch.tensor(_EDGES, device=cells.device)]
    crossing = inside[edges[..., 0]] != inside[edges[..., 1]]
    shared, inverse = torch.unique(edges[crossing].sort(-1).values, dim=0, return_inverse=True)
    f0, f1 = signed_values[shared[:, 0]], signed_values[shared[:, 1]]
    p0, p1 = points[shared[:, 0]], points[shared[:, 1]]
    vertices = (f0[:, None] * p1 - f1[:, None] * p0) / (f0 - f1)[:, None]
    edge_indices = cells.new_full((len(cells), 6), -1)
    edge_indices[crossing] = inverse
    mask = (signs.to(torch.long) * cells.new_tensor([1, 2, 4, 8])).sum(-1)
    cases = cells.new_tensor(_FACES)[mask]
    faces = edge_indices.gather(1, cases.clamp_min(0)).reshape(-1, 3)
    valid_faces = cases.reshape(-1, 3)[:, 0] >= 0
    faces = faces[valid_faces]
    # Select winding using the inside-to-outside direction of each tetrahedron.
    sites = points[cells]
    n_inside = signs.sum(-1)[:, None]
    inward = (sites * signs[..., None]).sum(1) / n_inside
    outward = (sites * ~signs[..., None]).sum(1) / (4 - n_inside) - inward
    outward = outward[:, None, :].expand(-1, 2, -1).reshape(-1, 3)[valid_faces]
    triangles = vertices[faces]
    normal = torch.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0], dim=-1)
    flip = (normal * outward).sum(-1) < 0
    faces = torch.where(flip[:, None], faces[:, [0, 2, 1]], faces)
    return PivotMesh(vertices, faces, shared)


def gaussian_pivot_mesh(means, quaternions, log_scales, raw_sdf, cells, *, pivot_ids=None):
    """Full continuous chain; independent raw SDF has nine entries per pivot."""
    pivots = gaussian_pivots(means, quaternions, log_scales, pivot_ids=pivot_ids)
    if raw_sdf.shape != pivots.shape[:2]:
        raise ValueError("nine independently learnable raw SDF values required per selected Gaussian")
    return marching_tetrahedra(pivots.reshape(-1, 3), cells, raw_sdf.tanh().reshape(-1))
