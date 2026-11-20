"""Owned, representation-neutral CPU geometry finishing.

Explicit NumPy arrays; no training/runtime imports or restricted geometry SDKs.
Native exact predicates are required for geometry acceptance, never substituted
by an approximate collision query. See specs/mesh/SEMANTICS.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from threading import local
from collections import deque
import time
import numpy as np

_operation_counts = local()


def _count_operation(name):
    counts = getattr(_operation_counts, 'values', None)
    if counts is None:
        counts = _operation_counts.values = {}
    counts[name] = counts.get(name, 0) + 1


def activation_counts():
    """Thread-local runtime observations; no input or result data is stored."""
    return dict(getattr(_operation_counts, 'values', {}))


class GeometryError(RuntimeError):
    """An operation could not meet its geometric postconditions."""

    def __init__(self, message, *, candidate=None, report=None):
        super().__init__(message)
        self.candidate = candidate
        self.report = report


def _native():
    try:
        from . import _geometry_native
    except ImportError as exc:
        raise ImportError("Build geometry with python tools/build_geometry.py (pybind11 and a C++17 compiler required)") from exc
    return _geometry_native


@dataclass(frozen=True)
class TriangleMesh:
    vertices: np.ndarray
    faces: np.ndarray

    def __post_init__(self):
        for name, value, dtype in (("vertices", self.vertices, np.float64), ("faces", self.faces, np.int64)):
            if not isinstance(value, np.ndarray) or value.dtype != dtype or value.ndim != 2 or value.shape[1] != 3 or not value.flags.c_contiguous:
                raise TypeError(f"{name} must be a C-contiguous NumPy {np.dtype(dtype)} (N,3) array; convert explicitly")
        if not np.isfinite(self.vertices).all():
            raise ValueError("vertices must be finite")
        if len(self.faces) and (self.faces.min() < 0 or self.faces.max() >= len(self.vertices)):
            raise ValueError("triangle indices out of bounds")


def _mesh(v, f):
    return TriangleMesh(np.ascontiguousarray(v, dtype=np.float64), np.ascontiguousarray(f, dtype=np.int64).reshape(-1, 3))


def compact(mesh):
    """Remove only unreferenced vertices. Does not weld or repair on import."""
    ids, inverse = np.unique(mesh.faces, return_inverse=True)
    return _mesh(mesh.vertices[ids], inverse.reshape(-1, 3))


def _edges(mesh):
    f = mesh.faces
    directed = np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]))
    keys = np.sort(directed, axis=1)
    order = np.lexsort((keys[:, 1], keys[:, 0]))
    ordered = keys[order]
    start = np.r_[0, np.flatnonzero(np.any(ordered[1:] != ordered[:-1], axis=1)) + 1] if len(order) else np.empty(0, np.int64)
    counts = np.diff(np.r_[start, len(order)])
    face_ids = np.tile(np.arange(len(f)), 3)
    two = start[counts == 2]
    pair_slots = np.column_stack((order[two], order[two + 1]))
    pairs = np.ascontiguousarray(face_ids[pair_slots])
    opposite = np.any(directed[pair_slots[:, 0]] != directed[pair_slots[:, 1]], axis=1)
    return directed, order, start, counts, pairs, opposite


def _components(mesh, pairs=None):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    if pairs is None:
        pairs = _edges(mesh)[4]
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(mesh.faces),) * 2).tocsr()
    return connected_components(graph, directed=False)


def boundary_loops(mesh):
    """Oriented simple indexed boundary cycles; branches are rejected."""
    directed, order, start, count, _, _ = _edges(mesh)
    if np.any(count > 2):
        raise GeometryError("nonmanifold edge: split/repair explicitly before filling")
    boundary = directed[order[start[count == 1]]]
    if not len(boundary):
        return []
    if len(np.unique(boundary[:, 0])) != len(boundary) or len(np.unique(boundary[:, 1])) != len(boundary):
        raise GeometryError("boundary branches at a multi-hole vertex")
    nxt = dict(boundary.tolist())
    if set(nxt) != set(nxt.values()):
        raise GeometryError("boundary is not a union of oriented cycles")
    loops = []
    while nxt:
        first = next(iter(nxt)); loop = [first]; current = nxt.pop(first)
        while current != first:
            loop.append(current); current = nxt.pop(current)
        if len(loop) < 3:
            raise GeometryError("boundary cycle has fewer than three vertices")
        loops.append(np.asarray(loop, np.int64))
    return loops


def split_vertex_fans(mesh):
    """Duplicate a vertex for each disconnected incident face fan.

    Coordinates remain exact duplicates. This fixes indexed topology only;
    coincident unshared contacts still fail geometric intersection acceptance.
    """
    _, _, _, counts, pairs, _ = _edges(mesh)
    if np.any(counts > 2):
        raise GeometryError("split_vertex_fans does not split nonmanifold edges")
    roots = _native().corner_components(mesh.faces, pairs).ravel()
    corners = mesh.faces.ravel()
    keys = np.column_stack((corners, roots))
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    original, first = np.unique(unique[:, 0], return_index=True)
    remap = np.empty(len(unique), np.int64)
    is_first = np.zeros(len(unique), bool); is_first[first] = True
    remap[first] = original
    extras = unique[~is_first, 0]
    remap[~is_first] = np.arange(len(mesh.vertices), len(mesh.vertices) + len(extras))
    return _mesh(np.concatenate((mesh.vertices, mesh.vertices[extras])), remap[inverse].reshape(-1, 3))


def clean_faces(mesh, *, degenerates=True, duplicates=True):
    """Explicitly remove exact degeneracies/duplicate indexed triangles.

    Returns (mesh, removed original face indices). No welding or vertex motion.
    """
    keep = np.ones(len(mesh.faces), bool)
    if degenerates:
        keep[_native().degenerate_faces(mesh.vertices, mesh.faces)] = False
    if duplicates:
        _, first = np.unique(np.sort(mesh.faces, axis=1), axis=0, return_index=True)
        unique = np.zeros(len(keep), bool); unique[first] = True; keep &= unique
    return _mesh(mesh.vertices, mesh.faces[keep]), np.flatnonzero(~keep)


def self_intersections(mesh, *, max_pairs=0, backend='cpu'):
    """Exact binary64 predicates, BVH broad phase. 0 means an exhaustive search."""
    if not isinstance(max_pairs, (int, np.integer)) or max_pairs < 0:
        raise ValueError("max_pairs must be a nonnegative integer")
    if backend == 'cuda_filtered_exact':
        from .cuda_ops import self_intersections_cuda
        return self_intersections_cuda(mesh, max_pairs=int(max_pairs))
    if backend != 'cpu': raise ValueError('unknown intersection backend')
    _count_operation('self_intersections_cpu')
    return _native().intersections(mesh.vertices, mesh.faces, int(max_pairs))


def audit(mesh, *, intersections=True, intersection_backend='cpu'):
    """Full edge/link audit and optional exhaustive geometric intersection audit."""
    start_time = time.perf_counter()
    d, order, start, counts, pairs, opposite = _edges(mesh)
    component_count, labels = _components(mesh, pairs)
    tri = mesh.vertices[mesh.faces]
    degenerate = _native().degenerate_faces(mesh.vertices, mesh.faces)
    duplicate = len(mesh.faces) - len(np.unique(np.sort(mesh.faces, axis=1), axis=0))
    bad_vertices = _native().vertex_links(mesh.faces, pairs, len(mesh.vertices))
    volumes = np.zeros(component_count)
    # A local reference reduces cancellation from large world translations.
    if len(tri):
        _, first = np.unique(labels, return_index=True)
        p = tri - tri[first, 0][labels, None, :]
        contribution = np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2]))
        volumes = np.bincount(labels, weights=contribution, minlength=component_count) / 6
    collision = self_intersections(mesh, backend=intersection_backend) if intersections else None
    closed = len(mesh.faces) > 0 and bool(np.all(counts == 2))
    topology = closed and not duplicate and not len(degenerate) and not len(bad_vertices) and bool(opposite.all())
    result = {
        "vertices": len(mesh.vertices), "faces": len(mesh.faces), "edges": len(counts),
        "components": int(component_count), "euler": int(len(np.unique(mesh.faces)) - len(counts) + len(mesh.faces)),
        "boundary_edges": int(np.sum(counts == 1)), "nonmanifold_edges": int(np.sum(counts > 2)),
        "nonmanifold_vertices": bad_vertices.tolist(), "degenerate_faces": degenerate.tolist(),
        "duplicate_faces": int(duplicate), "winding_consistent": bool(opposite.all()),
        "watertight_manifold": bool(topology), "component_volumes": volumes.tolist(),
        "outward": bool(topology and len(volumes) and np.all(volumes > 0)),
        "intersection_check": "exact_binary64_dyadic" if collision is not None else "not_checked",
        "self_intersection_pairs": len(collision["pairs"]) if collision is not None else None,
        "intersection_search_complete": bool(collision["complete"]) if collision is not None else False,
        "narrow_phase_pairs": int(collision["narrow_phase_pairs"]) if collision is not None else None,
    }
    result["passed"] = bool(result["outward"] and collision is not None and collision["complete"] and not len(collision["pairs"]))
    result['intersection_backend'] = intersection_backend if intersections else None
    if collision is not None and 'gpu_pruned_pairs' in collision:
        result['gpu_pruned_pairs'] = int(collision['gpu_pruned_pairs'])
        result['aabb_pairs'] = int(collision['aabb_pairs'])
    result["seconds"] = time.perf_counter() - start_time
    return result


def largest_component(mesh):
    """Keep the edge-connected component with greatest surface area."""
    _count_operation('component_selection')
    n, labels = _components(mesh)
    if not n:
        return compact(mesh)
    tri = mesh.vertices[mesh.faces]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    chosen = np.argmax(np.bincount(labels, weights=area, minlength=n))
    return compact(_mesh(mesh.vertices, mesh.faces[labels == chosen]))


def orient(mesh, *, outward=True):
    """Orient each component consistently. Closed components may be flipped out."""
    _, _, _, counts, pairs, opposite = _edges(mesh)
    if np.any(counts > 2):
        raise GeometryError("cannot orient nonmanifold edges")
    from scipy.sparse import coo_matrix
    rows = np.r_[pairs[:, 0], pairs[:, 1]]; cols = np.r_[pairs[:, 1], pairs[:, 0]]
    relation = np.r_[~opposite, ~opposite].astype(np.int8) + 1
    graph = coo_matrix((relation, (rows, cols)), shape=(len(mesh.faces),) * 2).tocsr()
    flip = np.full(len(mesh.faces), -1, np.int8)
    for seed in range(len(flip)):
        if flip[seed] >= 0: continue
        flip[seed] = 0; queue = deque([seed])
        while queue:
            i = queue.popleft()
            for p in range(graph.indptr[i], graph.indptr[i + 1]):
                j = graph.indices[p]; value = flip[i] ^ (graph.data[p] - 1)
                if flip[j] < 0: flip[j] = value; queue.append(j)
                elif flip[j] != value: raise GeometryError("nonorientable component")
    f = mesh.faces.copy(); f[flip == 1] = f[flip == 1, ::-1]
    candidate = _mesh(mesh.vertices, f)
    if outward:
        n, labels = _components(candidate)
        _, _, _, c, _, _ = _edges(candidate)
        if not np.all(c == 2):
            raise GeometryError("outward orientation requires closed components")
        tri = candidate.vertices[candidate.faces]
        for i in range(n):
            selected = labels == i; p = tri[selected] - tri[selected][0, 0]
            volume = np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2])).sum() / 6
            if volume == 0: raise GeometryError("zero-volume closed component")
            if volume < 0: f[selected] = f[selected, ::-1]
        candidate = _mesh(mesh.vertices, f)
    return candidate


def _plane(normal, offset):
    normal = np.asarray(normal, dtype=np.float64)
    if normal.shape != (3,) or not np.isfinite(normal).all() or not np.isfinite(offset):
        raise ValueError("plane requires a finite normal and offset")
    length = np.linalg.norm(normal)
    if not np.isfinite(length) or length == 0:
        raise ValueError("plane normal must have finite nonzero length")
    return normal / length, float(offset)


def _basis(normal):
    axis = np.eye(3)[np.argmin(np.abs(normal))]
    u = np.cross(axis, normal); u /= np.linalg.norm(u)
    return np.column_stack((u, np.cross(normal, u)))


def _in_polygon(point, polygon):
    x, y = point
    a = polygon; b = np.roll(polygon, -1, axis=0)
    mask = (a[:, 1] > y) != (b[:, 1] > y)
    crossings = a[mask, 0] + (y - a[mask, 1]) * (b[mask, 0] - a[mask, 0]) / (b[mask, 1] - a[mask, 1])
    return bool(np.count_nonzero(crossings > x) % 2)


def _triangulate_rings(vertices, rings, normal):
    """Owned exact-sign polygon triangulation, including nested hole bridges."""
    ids = np.concatenate(rings)
    axis = int(np.argmax(np.abs(normal)))
    coordinates = [(axis + 1) % 3, (axis + 2) % 3]
    points = vertices[ids][:, coordinates]
    local_vertices = np.ascontiguousarray(vertices[ids])
    boundaries = np.split(np.arange(len(ids)), np.cumsum([len(r) for r in rings])[:-1])
    try:
        indices = _native().triangulate(local_vertices, [r.tolist() for r in boundaries], axis)
    except RuntimeError as exc:
        raise GeometryError(str(exc)) from exc
    triangles = indices.tolist()
    # Ear slicing may omit exactly collinear vertices. Restore those vertices
    # by splitting cap edges; otherwise the side wall would contain T-junctions.
    used = set(indices.ravel().tolist())
    for p in range(len(points)):
        if p in used: continue
        found = False
        for i, triangle in enumerate(triangles):
            for j in range(3):
                a, b, c = triangle[j], triangle[(j + 1) % 3], triangle[(j + 2) % 3]
                edge = points[b] - points[a]; length2 = np.dot(edge, edge)
                if length2 == 0: continue
                t = np.dot(points[p] - points[a], edge) / length2
                sign = _native().projected_signs(local_vertices, np.array([[a, b, p]], np.int64), axis)[0]
                if 0 < t < 1 and sign == 0:
                    triangles[i] = [a, p, c]; triangles.append([p, b, c]); found = True; break
            if found: break
        if not found:
            raise GeometryError("triangulator omitted a boundary vertex that cannot be restored")
    f = ids[np.asarray(triangles, np.int64).reshape(-1, 3)]
    signs = _native().projected_signs(vertices, np.ascontiguousarray(f), axis) * np.sign(normal[axis])
    if np.any(signs == 0) or not len(f):
        raise GeometryError("cap triangulation has zero-area faces")
    f[signs < 0] = f[signs < 0, ::-1]
    # Every requested boundary edge must appear once, and every interior twice.
    test = _mesh(vertices, f)
    d, order, start, count, _, opposite = _edges(test)
    actual = np.sort(d[order[start[count == 1]]], axis=1)
    wanted = np.sort(np.concatenate([np.column_stack((r, np.roll(r, -1))) for r in rings]), axis=1)
    lex = lambda a: a[np.lexsort((a[:, 1], a[:, 0]))]
    if np.any(count > 2) or not opposite.all() or not np.array_equal(lex(actual), lex(wanted)):
        raise GeometryError("polygon triangulation did not preserve its exact indexed boundary")
    return f


def cap_boundaries(mesh, normal, offset, *, plane_tolerance=None):
    """Cap only boundary loops on a given plane, preserving nested holes."""
    n, offset = _plane(normal, offset)
    scale = max(np.ptp(mesh.vertices, axis=0).max(initial=0), abs(offset), np.abs(mesh.vertices).max(initial=0), 1)
    tolerance = scale * 64 * np.finfo(float).eps if plane_tolerance is None else plane_tolerance
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("plane_tolerance must be finite and nonnegative")
    loops = [r for r in boundary_loops(mesh) if np.all(np.abs(mesh.vertices[r] @ n - offset) <= tolerance)]
    if not loops: return mesh
    basis = _basis(n); origin = mesh.vertices[loops[0][0]]
    projected = [(mesh.vertices[r] - origin) @ basis for r in loops]
    areas = [abs(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - p[:, 1] * np.roll(p[:, 0], -1))) for p in projected]
    parent = np.full(len(loops), -1, np.int64)
    for i, polygon in enumerate(projected):
        containing = [j for j, other in enumerate(projected) if areas[j] > areas[i] and _in_polygon(polygon[0], other)]
        if containing: parent[i] = min(containing, key=lambda j: areas[j])
    depths = []
    for i in range(len(loops)):
        depth = 0; p = parent[i]
        while p >= 0: depth += 1; p = parent[p]
        depths.append(depth)
    patches = []
    for i, depth in enumerate(depths):
        if depth % 2: continue
        rings = [loops[i]] + [loops[j] for j in range(len(loops)) if parent[j] == i]
        patches.append(_triangulate_rings(mesh.vertices, rings, -n))
    result = _mesh(mesh.vertices, np.concatenate([mesh.faces, *patches]))
    if not _edges(result)[5].all():
        raise GeometryError("cap orientation conflicts with the existing boundary")
    return result


def clip_plane(mesh, normal=(0., 0., 1.), offset=0., *, cap=True):
    """Keep the positive halfspace. Share every generated source-edge vertex."""
    _count_operation('cpu_clip')
    n, offset = _plane(normal, offset)
    v, f = mesh.vertices, mesh.faces
    distance = v @ n - offset
    positive = distance[f] >= 0
    keep = positive.all(axis=1)
    cross_faces = f[positive.any(axis=1) & ~keep]
    triangles = f[keep].tolist()
    additions = []; cuts = {}; axis = int(np.argmax(np.abs(n)))

    def cut(a, b):
        if distance[a] == 0: return int(a)
        if distance[b] == 0: return int(b)
        key = (min(int(a), int(b)), max(int(a), int(b)))
        if key not in cuts:
            a, b = key
            t = distance[a] / (distance[a] - distance[b])
            p = v[a] + t * (v[b] - v[a])
            other = [k for k in range(3) if k != axis]
            p[axis] = (offset - np.dot(p[other], n[other])) / n[axis]
            cuts[key] = len(v) + len(additions); additions.append(p)
        return cuts[key]

    for face in cross_faces:
        polygon = []
        for j in range(3):
            a, b = face[j], face[(j + 1) % 3]
            if distance[a] >= 0: polygon.append(int(a))
            if (distance[a] < 0 < distance[b]) or (distance[b] < 0 < distance[a]): polygon.append(cut(a, b))
        if len(polygon) >= 3:
            for j in range(1, len(polygon) - 1): triangles.append([polygon[0], polygon[j], polygon[j + 1]])
    vertices = np.concatenate((v, np.asarray(additions).reshape(-1, 3))) if additions else v
    clipped = _mesh(vertices, triangles)
    return cap_boundaries(clipped, n, offset) if cap and len(clipped.faces) else clipped


def fill_holes(mesh, *, max_edges=None, check=True, method='projected', intersection_backend='cpu'):
    """Patch simple boundary loops, without moving any original vertex.

    Nonplanar loops are explicitly projected onto their best-fit plane for
    triangulation, then lifted using their original vertices. Whole-mesh
    intersection/topology acceptance is required by default.
    """
    _count_operation('boundary_patches')
    if method not in ('projected', 'minimal_area'):
        raise ValueError("hole method must be projected or minimal_area")
    if method == 'minimal_area' and max_edges is None:
        raise ValueError("minimal_area requires an explicit max_edges bound")
    loops = boundary_loops(mesh); patches = []
    forbidden = np.empty((0, 2), np.int64)
    if method == 'minimal_area' and loops:
        loop_id = np.full(len(mesh.vertices), -1, np.int64)
        for i, loop in enumerate(loops): loop_id[loop] = i
        edges = _edges(mesh)[0]
        same = (loop_id[edges[:, 0]] >= 0) & (loop_id[edges[:, 0]] == loop_id[edges[:, 1]])
        forbidden = np.ascontiguousarray(np.unique(np.sort(edges[same], axis=1), axis=0))
    if max_edges is not None and (not isinstance(max_edges, int) or max_edges < 3):
        raise ValueError("max_edges must be >=3 or None")
    for loop in loops:
        if max_edges is not None and len(loop) > max_edges:
            raise GeometryError(f"hole has {len(loop)} edges, exceeds max_edges")
        if method == 'minimal_area':
            try:
                patches.append(_native().patch_3d(mesh.vertices, loop, forbidden, max_edges))
            except RuntimeError as exc:
                raise GeometryError(str(exc)) from exc
            continue
        points = mesh.vertices[loop]; centered = points - points.mean(axis=0)
        _, _, basis = np.linalg.svd(centered, full_matrices=False)
        n = basis[-1]
        direction = np.sum(np.cross(centered, np.roll(centered, -1, axis=0)), axis=0)
        if np.dot(n, direction) > 0: n = -n
        patches.append(_triangulate_rings(mesh.vertices, [loop], n))
    if not patches:
        candidate = mesh
    else:
        candidate = _mesh(mesh.vertices, np.concatenate([mesh.faces, *patches]))
    candidate = orient(candidate)
    if check:
        report = audit(candidate, intersection_backend=intersection_backend)
        if not report["passed"]: raise GeometryError(f"hole patches failed validation: {report}")
    return candidate


def smooth_region(mesh, movable, *, max_displacement, max_backtracks=12):
    """Uniform harmonic patch solve with locked exterior and global validation."""
    from scipy.sparse import coo_matrix, diags
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse.linalg import spsolve
    if not isinstance(movable, np.ndarray) or movable.dtype != bool or movable.shape != (len(mesh.vertices),):
        raise TypeError("movable must be a NumPy bool vertex mask")
    if not np.isfinite(max_displacement) or max_displacement < 0 or max_backtracks < 0:
        raise ValueError("invalid displacement/backtrack bound")
    baseline = audit(mesh)
    if not baseline["passed"]: raise GeometryError("smoothing requires an accepted input mesh")
    if not movable.any(): return mesh
    f = mesh.faces
    edges = np.unique(np.sort(np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]])), axis=1), axis=0)
    rows = np.r_[edges[:, 0], edges[:, 1]]; cols = np.r_[edges[:, 1], edges[:, 0]]
    adjacency = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(mesh.vertices),) * 2).tocsr()
    _, labels = connected_components(adjacency, directed=False)
    if not np.isin(np.unique(labels[movable]), np.unique(labels[~movable])).all():
        raise GeometryError("each movable component needs a fixed boundary anchor")
    laplacian = diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency
    free = np.flatnonzero(movable); fixed = np.flatnonzero(~movable)
    solved = spsolve(laplacian[free][:, free].tocsc(), -(laplacian[free][:, fixed] @ mesh.vertices[fixed]))
    delta = solved - mesh.vertices[free]
    lengths = np.linalg.norm(delta, axis=1)
    delta *= np.minimum(1, max_displacement / np.maximum(lengths, np.finfo(float).tiny))[:, None]
    for step in range(max_backtracks + 1):
        v = mesh.vertices.copy(); v[free] += delta * 2. ** -step
        candidate = _mesh(v, mesh.faces)
        if audit(candidate)["passed"]: return candidate
    raise GeometryError("no valid smoothing step within the bounded backtracking budget")


def repair_self_intersections(mesh, *, max_iterations=3, expand_rings=0, max_removed_faces=10000, hole_method='projected', max_hole_edges=None, keep_largest=False, intersection_backend='cpu'):
    """Bounded local removal and boundary retriangulation; no global remesh."""
    _count_operation('local_patch_prefix')
    if max_iterations < 1 or expand_rings < 0 or max_removed_faces < 1:
        raise ValueError("invalid repair budget")
    current = mesh; removed_total = 0
    for _ in range(max_iterations):
        collision = self_intersections(current, backend=intersection_backend)
        if not len(collision["pairs"]):
            candidate = fill_holes(current, method=hole_method, max_edges=max_hole_edges, intersection_backend=intersection_backend)
            if audit(candidate, intersection_backend=intersection_backend)["passed"]: return candidate
        bad = np.zeros(len(current.faces), bool)
        bad[collision["pairs"].ravel()] = True
        pairs = _edges(current)[4]
        for _ring in range(expand_rings):
            bad[pairs[bad[pairs].any(axis=1)].ravel()] = True
        removed_total += int(bad.sum())
        if removed_total > max_removed_faces:
            raise GeometryError(f"local repair needs {removed_total} removed faces, exceeds total budget")
        if bad.all(): raise GeometryError("repair would delete the whole surface")
        candidate = split_vertex_fans(_mesh(current.vertices, current.faces[~bad]))
        if keep_largest: candidate = largest_component(candidate)
        current = fill_holes(candidate, check=False, method=hole_method, max_edges=max_hole_edges)
        report = audit(current, intersection_backend=intersection_backend)
        if report["passed"]: return current
    report = audit(current, intersection_backend=intersection_backend)
    raise GeometryError(f"local cut-and-patch repair did not converge: {report}", candidate=current, report=report)


def repair_contact_stars(mesh, *, max_iterations=3, expand_rings=1, max_removed_faces=10000, max_hole_edges=2048, intersection_backend='cpu', geometric_contacts=False):
    """Repair geometric pinch contacts by excising complete singular stars.

    Exact coordinate welding changes indices only. The full vertex link, not
    only a colliding triangle pair, defines the region to remove. Original
    coordinates are retained, and no global field or smoothing is introduced.
    Every result still requires exhaustive geometric acceptance.
    """
    _count_operation('contact_star_repair')
    if max_iterations < 1 or expand_rings < 0 or max_removed_faces < 1:
        raise ValueError('invalid contact repair budget')
    current = mesh; removed_total = 0; report = None
    for _ in range(max_iterations):
        vertices, inverse = np.unique(current.vertices, axis=0, return_inverse=True)
        welded = _mesh(vertices, inverse[current.faces])
        topology = audit(welded, intersections=False)
        collision = self_intersections(welded, backend=intersection_backend)
        if topology['outward'] and not len(collision['pairs']):
            return compact(welded)
        directed, order, start, counts, neighbors, _ = _edges(welded)
        singular = np.asarray(topology['nonmanifold_vertices'], np.int64)
        # A manifold vertex link can still fold geometrically. Remove the
        # complete shared-vertex star, not only the colliding face pair.
        shared_contacts = []
        if geometric_contacts:
            for a, b in collision['pairs']:
                shared_contacts.extend(set(welded.faces[a]).intersection(welded.faces[b]))
        if shared_contacts:
            singular = np.unique(np.r_[singular, shared_contacts])
        if np.any(counts > 2):
            singular = np.unique(np.r_[singular, directed[order[start[counts > 2]]].ravel()])
        bad = np.isin(welded.faces, singular).any(axis=1)
        bad[collision['pairs'].ravel()] = True
        bad[np.asarray(topology['degenerate_faces'], np.int64)] = True
        for _ring in range(expand_rings):
            bad[neighbors[bad[neighbors].any(axis=1)].ravel()] = True
        removed_total += int(bad.sum())
        if removed_total > max_removed_faces or bad.all():
            raise GeometryError('contact-star removal exceeds the bounded local region', candidate=current)
        opened = _mesh(welded.vertices, welded.faces[~bad])
        opened = largest_component(split_vertex_fans(opened))
        try:
            current = fill_holes(opened, check=False, method='minimal_area', max_edges=max_hole_edges)
        except GeometryError as exc:
            raise GeometryError(f'contact-star patch: {exc}', candidate=opened) from exc
        report = audit(current, intersection_backend=intersection_backend)
        if report['passed']:
            return current
    raise GeometryError(f'contact-star repair did not converge: {report}', candidate=current, report=report)


def refine_face_count(mesh, target):
    """Reach an exact attainable count by coplanar centroid face splits.

    Each split adds two triangles. This preserves the represented surface and
    creates no detail. Exact binary64 constructions are not promised.
    """
    if not isinstance(target, (int, np.integer)) or target < len(mesh.faces) or (target - len(mesh.faces)) % 2:
        raise ValueError("target must be >= current faces with an even difference")
    v, f = mesh.vertices.copy(), mesh.faces.copy()
    if not len(f) and target: raise ValueError("cannot refine an empty mesh")
    while len(f) < target:
        count = min((target - len(f)) // 2, len(f))
        tri = v[f]
        area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        selected = np.argpartition(area, len(area) - count)[-count:]
        chosen = f[selected]; ids = np.arange(len(v), len(v) + count)
        new = np.concatenate([np.column_stack((chosen[:, j], chosen[:, (j + 1) % 3], ids)) for j in range(3)])
        keep = np.ones(len(f), bool); keep[selected] = False
        v = np.concatenate((v, tri[selected].mean(axis=1))); f = np.concatenate((f[keep], new))
    return _mesh(v, f)
