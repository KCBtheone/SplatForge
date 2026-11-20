# Mesh processing v1 (2026-10-20)

This CPU geometry layer implements clipping, patching, and finishing above
the operator runtime. Inputs are explicit
NumPy float64 vertices (N,3), int64 triangle indices (M,3), contiguous; no
implicit Torch detach, device move, repair on import, or float32 quantization.
Topology changes are discrete and not differentiable. No MeshLib/CGAL code,
headers, binaries or runtime calls are implementation inputs.

Operations: topology/vertex-link audit, exact-predicate triangle intersections
with BVH broad phase, edge-connected component selection, consistent/outward
orientation, shared-edge plane clipping, planar caps including nested rings,
projected boundary patches, bounded local cut-and-patch repair, fixed-boundary
sparse harmonic smoothing, surface-preserving face refinement and optional
explicit QEM simplification. Existing vertices outside edited regions stay
bit-identical. No whole-surface voxelization is implicit.

`minimal_area` is an explicitly selected, bounded 3D patch algorithm for folded
nonplanar loops. Its O(n^3) dynamic program forbids diagonals already present in
the exterior surface. It is not substituted silently for projected patches.
Topological triangulation does not imply embedding; the whole-mesh exact audit
remains mandatory. Limits are caller-specified. Degenerate/duplicate cleaning
and disconnected fan splitting are separate explicit operations.

Intersection semantics: two different triangles must meet only in their shared
indexed simplex. Other contacts, coplanar overlaps, coincident but unshared
vertices/edges and duplicate triangles count. An incomplete pair search cannot
pass an audit. Orientation signs refer to exact arithmetic on binary64 inputs;
filtered native predicates fall back to owned arbitrary-precision dyadic integer
arithmetic (not an epsilon test). Degenerate faces are reported separately.

Clip keeps dot(x, unit_normal) >= offset. Offset is world distance. On-plane
vertices are shared; no tolerance-based weld. Plane intersection coordinates
are binary64 constructions, not rational-exact constructions. Coplanar faces
are retained. Capping checks the resulting topology and orientation, and cannot
silently discard collinear boundary vertices. Each cap has an explicit plane;
nested contours alternate solid/hole. A projected nonplanar hole fill is an
explicit approximation: patch faces are checked against the entire mesh.

Repair removes only intersecting-face neighborhoods within explicit iteration
and face-count limits, patches their boundaries, and returns only after a
complete audit. Unsupported/nonconvergent cases raise with diagnostics;
arbitrary self-intersecting soups are not promised to be repairable. Orientation
does not pretend to solve nesting or remove internal disconnected shells.
Largest component is explicit, selected by area. Smoothing fixes every vertex
outside its mask and requires an anchored component; movement is bounded and
backtracked until a full geometry audit passes.

Validation must cover analytic concave caps, holes, shared clipping edges,
coplanar overlap, adjacent folds, bow-tie vertices, near-coplanar predicates,
an exact rational predicate oracle, and representative reconstructed meshes.
Record CPU/platform, face count, complete audit, and per-stage timings.
