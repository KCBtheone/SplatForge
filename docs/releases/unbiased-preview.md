# Unbiased Surfel and PGSR-style auxiliary operators

This is an independent experimental companion to SplatForge. Its historical
package name is `commercial-unbiased-gsplat`; the name is not a certification
of a complete commercial runtime environment.

The package implements paper-derived corrected surfel depth, depth convergence,
depth normals, PGSR-style plane-homography consistency, patch ZNCC, and planar
Gaussian helpers. It is not a registered SplatForge Studio method and does not
provide a complete public reproduction of either paper.

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
`setup.py` builds the optional native extension when CUDA is available. On a
CPU installation, the Torch implementation remains available.

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
