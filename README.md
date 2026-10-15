# SplatForge

**An operator-first research prototype for geometry-oriented Gaussian Splatting.**

SplatForge explores how components from Gaussian Splatting papers can become
reusable, testable operators. A small reconstruction framework provides a place
to exercise those operators through shared data loading, training, checkpoints,
evaluation, mesh export, and a local viewer.

**Status: early prototype (`0.0.1a1`).** APIs and abstractions may change
substantially. This preview makes the current implementation available while
development continues. Paper benchmark parity, production readiness, and
arbitrary cross-method compatibility have not been established.

## What is included

| Component | Current scope |
| --- | --- |
| Operator runtime | Versioned contracts, provider registration, explicit capabilities, dispatch explanations, and conformance tooling |
| Gaussian operators | Covariance, ray intersections, spherical harmonics, depth normals, alpha compositing, depth distortion, color ambiguity, opacity fields, and patch ZNCC |
| Surface Studio | Experimental RaDe-GS, AmbiSuR, and GOF recipes sharing a training and export workflow |
| Independent companion | Unbiased Surfel and PGSR-style auxiliary operators in [`commercial_unbiased_gsplat/`](commercial_unbiased_gsplat/); separately installed and not registered Studio methods |

RaDe-GS optionally uses a pinned external gsplat port. The Torch surface
operators and the AmbiSuR/GOF compositions are experimental implementations.
The companion includes its own source policy and tensor specification. The
preview contains source, tests, and documentation; experimental inputs,
checkpoints, scan outputs, and model weights are not part of the distribution.

## Install

Use Python 3.10 or later in a dedicated environment. Install a PyTorch build
appropriate for your machine first, then install the Studio dependencies:

```bash
git clone https://github.com/KCBtheone/SplatForge.git
cd SplatForge
python -m pip install -e '.[studio]'
splatforge methods
```

Release wheels are also available on the
[Releases page](https://github.com/KCBtheone/SplatForge/releases).
The default wheel contains Python code and CUDA source for an optional JIT
extension; it does not contain prebuilt CUDA binaries or third-party renderers.

## Try the workflow on CPU

```bash
splatforge demo --output demo-data --width 24 --points 48
sf-train gof --data demo-data --output outputs/demo --device cpu --backend torch \
  --steps 3 --set data.downscale=1 --set data.init_points=48 \
  --set model.opacity=0.7 --set densify.enabled=false \
  --set loss.geometric_start=0 --set train.export_on_finish=false
sf-export --run outputs/demo --device cpu \
  --set export.alpha_min=0.1 --set export.truncation=0.15
sf-viewer --run outputs/demo --device cpu
```

The demo starts from an analytic sphere and generated images. It checks the
workflow and file formats; it is not evidence of reconstruction accuracy.
The commands above use Bash line continuation. In PowerShell, put each command
on one line or use a backtick for continuation.

## Use an operator directly

```python
import torch
from splatforge.gaussian import SurfaceOperators

ops = SurfaceOperators()
quaternions = torch.tensor([[1.0, 0.0, 0.0, 0.0]], requires_grad=True)
scales = torch.tensor([[0.1, 0.2, 0.3]], requires_grad=True)
covariance = ops("gaussian.covariance@1", quaternions, scales)
covariance.square().sum().backward()
```

Operator contracts are recorded in
[`semantics.json`](src/splatforge/gaussian/semantics.json).
The runtime rejects unsupported calls explicitly; backend substitutions are
not hidden behind the API.

## Documentation and development

- [Studio guide](docs/studio/PUBLIC_GUIDE.md): inputs, backends, plugins, and exports.
- [Architecture](ARCHITECTURE.md): design direction; some sections describe planned work.
- [Roadmap](docs/ROADMAP.md): feasibility work, the first operator library, and later expansion.
- [Source and dependency notes](THIRD_PARTY.md).
- [Preview release notes](docs/releases/0.0.1a1.md).
- [Contributing](CONTRIBUTING.md).

For the source checks:

```bash
python -m pip install -e '.[studio,test]'
python tools/quality/run_phase0_checks.py
```

The `Phase 0` name refers to the feasibility stage and its local checks, not a
completed paper reproduction. CUDA backends require separate environment and
GPU validation. The current release verification is described in the release
notes.

## License

SplatForge's original source is licensed under [Apache-2.0](LICENSE).
Optional dependencies retain their own licenses and notices. See
[NOTICE](NOTICE) and [THIRD_PARTY.md](THIRD_PARTY.md) for attribution and the
scope of this source distribution.
