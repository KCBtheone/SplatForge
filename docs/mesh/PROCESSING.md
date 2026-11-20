# CPU mesh processing

`splatforge.mesh` is a representation-neutral geometry layer. It does not
import the trainer, add mesh concepts to `core`, or require CUDA. Its owned C++17
provider implements triangle BVH traversal, filtered orientation predicates
with exact floating expansions and dyadic arithmetic, polygon triangulation and quadric edge collapse.
NumPy/SciPy implement topology assembly, clipping, local patches and sparse
harmonic smoothing. MeshLib and CGAL are not imports, link dependencies, headers
or runtime tools of this layer.

## Install

Base SplatForge remains installable without a compiler. To build the geometry
provider into an installed package (C++17 compiler required):

```bash
SPLATFORGE_BUILD_GEOMETRY=1 python -m pip install '.[geometry]'
```

PowerShell:

```powershell
$env:SPLATFORGE_BUILD_GEOMETRY='1'
python -m pip install '.[geometry]'
```

For development from a source checkout:

```bash
python -m pip install -e '.[geometry,test]' pybind11
python tools/build_geometry.py
python -m pytest tests/integration/test_mesh_processing.py tests/integration/test_mesh_geometry_workflow.py
```

On Windows install Visual Studio Build Tools with the C++ workload. MSVC
discovery is handled by setuptools. Stop any process using the extension
before rebuilding it. On Linux use a C++17 compiler; no CUDA compiler is needed.
Never enable fast-math for the exact predicate translation unit.

The optional CUDA clipping backend installs with `.[geometry-cuda12]`.
It uses owned binary64 PTX through the CUDA driver, with CuPy handling memory
and launches. It does not require NVRTC or a CUDA Toolkit installation. The
GPU computes plane distances, triangle classifications and edge intersections;
indexed assembly, polygon capping and final geometry acceptance remain on CPU.
This is a hybrid geometry operation, not an all-GPU repair implementation.

The explicit `intersection_backend='cuda_filtered_exact'` adds a conservative
GPU separation stage to exact self-intersection queries. A bounded list of
inclusive triangle-AABB pairs is produced by the owned CPU BVH. Binary64 CUDA
projection tests discard only pairs with certified separation, allowing contact
only at shared indexed endpoints; uncertain pairs use the unchanged exact CPU
predicate. Its numerical filter bound is an arithmetic error bound, not a
geometric tolerance. Coplanar, coincident and extreme-coordinate cases can take
the exact refinement path. This is an intentional hybrid algorithm. There is
no fallback to the complete CPU query if CUDA is unavailable or the declared
AABB pair allocation budget is exceeded; the operation raises.

The distributed PTX needs only a compatible CUDA driver. Rebuilding it uses
`python tools/build_geometry_cuda.py` with nvcc on PATH, without fast math or
FMA contraction. No PyTorch extension rebuild is involved.

## Use

```python
from splatforge.mesh import FinishConfig, finish_mesh, audit
from splatforge.mesh.formats import read_mesh, write_mesh

mesh = read_mesh('input.ply')
mesh, report = finish_mesh(mesh, FinishConfig(
    cut_offset=0.025,       # world distance, in the input coordinate system
    target_faces=300_000,
))
write_mesh('finished.ply', mesh)  # double coordinates
assert audit(read_mesh('finished.ply'))['passed']
```

For an already upright mesh requiring the explicit two-plane local
repair flow:

```python
mesh, report = finish_mesh(mesh, FinishConfig.native_local(
    initial_cut=0., final_cut=.02, target_faces=300_000, clip_backend='cuda'))
```

To select GPU filtering throughout the recipe, including its final audit, pass
`intersection_backend='cuda_filtered_exact'` to `FinishConfig.native_local`.
The same option is available on `audit`; `self_intersections` calls it `backend`.

This named recipe performs an open trim, fan splitting and bounded 3D hole
patches, followed by a bounded cut/patch prefix and contact-star repair. The
prefix may leave an explicitly reported partial intermediate; the contact-star
stage always executes and only the final full audit can accept an output.
Exact-coordinate welding changes indices, without quantizing coordinates.
The final crop is followed by largest-component selection, base translation
and face-budget adjustment. Exceptions never turn a partial mesh into success.
The `.02` plane is a caller-supplied physical choice, not a guessed scale.

```bash
sf-mesh input.ply --audit-only
sf-mesh input.ply --output finished.ply --cut-offset 0.025 --target-faces 300000
# Also usable directly in a checkout:
python -m splatforge.mesh.cli input.ply --audit-only
python -m splatforge.mesh.cli input.ply --output cut.ply --cut-offset 0.025 --clip-backend cuda
python -m splatforge.mesh.cli upright.ply --output finished.ply --recipe native-local --clip-backend cuda
python -m splatforge.mesh.cli upright.ply --output gpu.ply --recipe native-local --clip-backend cuda --intersection-backend cuda_filtered_exact
```

Inputs to `TriangleMesh` are contiguous `float64` vertex and `int64` triangle
NumPy arrays. Conversions from Torch/device/dtype must be explicit. PLY/OFF I/O
does not silently merge vertices, reject selected triangles or downcast the
coordinates to float32. The existing Studio viewer writer is a separate API.

Individual operations:

| API | Semantics |
| --- | --- |
| `audit`, `self_intersections` | Edge/link topology, orientation, components, exact contact tests; bounded searches explicitly report incomplete |
| `boundary_loops` | Oriented simple boundary cycles; branched boundaries raise |
| `clean_faces` | Explicit exact-degenerate/duplicate face removal, with removed indices |
| `split_vertex_fans` | Duplicate disconnected indexed umbrellas; does not hide coincident geometric contacts |
| `largest_component` | Select by area, never silently within an individual audit |
| `orient` | Consistent component winding; outward direction for closed components |
| `clip_plane`, `cap_boundaries` | Shared-edge halfspace clipping and nested-ring planar caps |
| `clip_plane_cuda` | Explicit CUDA kernels for clipping, with CPU assembly/capping and separate kernel/end-to-end timings |
| `fill_holes` | Explicit projected or bounded 3D minimum-area patches with unchanged original coordinates |
| `repair_self_intersections` | Bounded local face removal and retriangulation; accepted only after full audit |
| `smooth_region` | Sparse harmonic solve, locked exterior, displacement bound and validated backtracking |
| `simplification.simplify` | Owned QEM, edge-link/normal/duplicate constraints, locked vertices, final exact audit |
| `refine_face_count` | Exact attainable count through coplanar centroid subdivision; creates no detail |

The finishing recipe selects the largest component, orients it, optionally
cuts/caps, fills boundaries, performs bounded local repair if needed, adjusts
the face budget, and runs full acceptance. Destructive cleaning/fan splitting
are explicit `FinishConfig` options and are off by default. A caller must
supply the upright coordinate transform; there is no guessed alignment.

## Correctness and failure behavior

Intersection predicates operate on the exact values of the provided binary64
coordinates. Indexed shared vertices/edges are legal contacts; any overlap
beyond them fails. Coincident *unshared* vertices/edges also count. Exact signs
do not make generated clipping/subdivision coordinates rational-exact.

An audit passes only for nonempty, closed, consistently outward manifold
geometry, with no exact degenerate/duplicate triangles, disconnected vertex
umbrellas or geometric self-intersections. Multiple outward components can
pass `audit`; the finishing recipe requires one if `keep_largest=True`.
Genus zero is not required. Outward-per-component orientation does not infer
the semantics of nested inner shells. Largest-component removal is explicit.

The complete self-intersection query is BVH accelerated and streams node pairs;
it never materializes an all-triangle pair matrix. A nonzero `max_pairs` is a
diagnostic early stop, not valid acceptance evidence. Exact degenerate faces
fail independently of intersection reporting.

`fill_holes(method='minimal_area', max_edges=2048)` explicitly selects a
three-dimensional minimum-area dynamic program for nonplanar/folded loops.
It forbids existing exterior diagonals, costs O(n^3) time/O(n^2) memory, and
requires a caller-specified size limit. A projected patch is the faster default.
Neither algorithm proves an embedded surface before the complete audit.

Projected nonplanar hole filling and local cut-and-patch repair are bounded
algorithms, not arbitrary-soup Boolean reconstruction. Large folded holes,
nonmanifold edges, unresolvable triangle contacts, impossible face targets or
an exhausted repair budget raise `GeometryError`; no success marker or altered
source file is produced. Whole-surface voxel resampling, generalized winding
offsets and alpha wrapping are not implicitly substituted.

QEM error is an accumulated unweighted squared plane-distance quadric, not a
Hausdorff guarantee. Collapses cannot remove locked vertices or move boundary
vertices; unreachable budgets raise. Full exact intersection acceptance can
reject a locally valid QEM candidate. Smoothing similarly backtracks or raises.

## Evidence

The [mesh semantics](../../specs/mesh/SEMANTICS.md) specify the operations and
acceptance checks. `tools/geometry_benchmark.py` measures processing time and
reports audit results for supplied meshes.
