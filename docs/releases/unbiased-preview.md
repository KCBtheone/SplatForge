# Unbiased Surfel and PGSR-style auxiliary operators

This is an independent experimental companion to SplatForge. Its historical
package name is `commercial-unbiased-gsplat`; the name is not a certification
of a complete commercial runtime environment.

The package implements paper-derived corrected surfel depth, depth convergence,
depth normals, PGSR-style plane-homography consistency, patch ZNCC, and planar
Gaussian helpers. The Studio `unbiased-surfel` recipe uses this package's
native CUDA extension.

## Install and test

Install PyTorch first. From this directory:

```bash
python -m pip install 'setuptools>=69' wheel
python -m pip install --no-deps --no-build-isolation -e .
python -m unittest discover -s tests -v
```

The optional gsplat integration expects
`nerfstudio-project/gsplat` at
`6f378361f32ba4f1be86f65ae0db4948efb37dd5`. It is installed separately.
The optional native extension builds automatically when PyTorch sees a CUDA
device. Set `UNBIASED_GSPLAT_BUILD_CUDA=0` for a Torch-only build or
`UNBIASED_GSPLAT_BUILD_CUDA=1` to require the extension. A native build needs
the CUDA toolchain. Build errors are reported by the installer.

## Scope and sources

`SPEC.md` defines tensor semantics and gradient boundaries.
`SOURCE_POLICY.md` records permitted and excluded implementation inputs.
`NOTICE` preserves the Unbiased paper attribution. The public source snapshot
contains original Python/CUDA source and synthetic tests. It excludes private
scan data, customer deliveries, model checkpoints, experiment records, and
source-free reference fixtures that have not been supplied for publication.

The separately licensed Unbiased and PGSR author implementations are not
bundled or runtime dependencies. Source-isolation is an engineering record;
it is not a legal or patent-clearance conclusion. CPU synthetic tests do not
establish reference equivalence, GPU acceptance, or reconstruction quality.
