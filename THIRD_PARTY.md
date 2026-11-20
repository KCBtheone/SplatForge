# Source and dependency notes

This preview distributes SplatForge source under Apache-2.0. It does not bundle
external renderers, model weights, datasets, or a frozen runtime environment.

| Component | Integration boundary |
| --- | --- |
| PyTorch, PyYAML, jsonschema | Base Python dependencies, installed separately |
| NumPy, SciPy, Pillow | Studio dependencies, installed separately |
| Open3D | Optional voxel TSDF backend, installed separately |
| pybind11 | BSD-3-Clause; build dependency of the optional owned CPU geometry extension |
| CuPy, NVIDIA CUDA runtime | Optional CUDA clipping launch/memory integration; installed separately, not bundled |
| `Mollylulu/gsplat_radegs` | Optional RaDe-GS backend at `23b0f82551a938a05b5b99df06edf220b9305a51`; not vendored |
| `nerfstudio-project/gsplat` | PGSR and Unbiased Surfel renderer dependency at `6f378361f32ba4f1be86f65ae0db4948efb37dd5`, installed separately |
| Depth Anything 3 | Optional external prior-generation integration; SDK and weights are not bundled |

Recorded source identities and implementation differences are in
[`provenance/surface_methods.json`](provenance/surface_methods.json).
Those records describe engineering provenance; they do not certify every
possible installed dependency environment.

The AmbiSuR, GOF, PGSR, and Unbiased author implementations are not bundled or
runtime dependencies of this preview. Original paper citations and the
companion's source-isolation rules are retained. The companion's package name
is historical and does not imply a complete commercial dependency review.
VCR-GauS uses SplatForge's native CUDA rasterizer. The VCR-GauS author code and
its rasterizer fork are outside the source distribution and runtime.

The main distribution contains optional CUDA source written for SplatForge.
The optional CPU mesh processing provider is also owned SplatForge source.
It uses NumPy/SciPy and build-time pybind11. MeshLib and CGAL are not runtime,
header or link dependencies of that provider. This describes the mesh module,
not every dependency installed in an unrelated reconstruction environment.
The companion includes its own optional extension source. Native builds and
their toolchains remain environment-specific; the wheel is not a prebuilt GPU
environment.
