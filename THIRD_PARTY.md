# Source and dependency notes

This preview distributes SplatForge source under Apache-2.0. It does not bundle
external renderers, model weights, datasets, or a frozen runtime environment.

| Component | Integration boundary |
| --- | --- |
| PyTorch, PyYAML, jsonschema | Base Python dependencies, installed separately |
| NumPy, SciPy, Pillow | Studio dependencies, installed separately |
| Open3D | Optional voxel TSDF backend, installed separately |
| `Mollylulu/gsplat_radegs` | Optional RaDe-GS backend at `23b0f82551a938a05b5b99df06edf220b9305a51`; not vendored |
| `nerfstudio-project/gsplat` | Companion integration at `6f378361f32ba4f1be86f65ae0db4948efb37dd5`; not vendored |
| Depth Anything 3 | Optional external prior-generation integration; SDK and weights are not bundled |

Recorded source identities and implementation differences are in
[`provenance/surface_methods.json`](provenance/surface_methods.json).
Those records describe engineering provenance; they do not certify every
possible installed dependency environment.

The AmbiSuR, GOF, PGSR, and Unbiased author implementations are not bundled or
runtime dependencies of this preview. Original paper citations and the
companion's source-isolation rules are retained. The companion's package name
is historical and does not imply a complete commercial dependency review.

The main distribution contains optional CUDA source written for SplatForge.
The companion includes its own optional extension source. Native builds and
their toolchains remain environment-specific; the wheel is not a prebuilt GPU
environment.
