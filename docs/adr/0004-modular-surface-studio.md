# ADR 0004: Modular surface studio above the operator runtime

- Status: experimental implementation
- Date: 2026-10-12

The first useful application includes RaDe-GS, AmbiSuR and GOF. The user
authorized direct integration of a permissively licensed gsplat RaDe-GS
implementation and requested a modular framework comparable in workflow to
SDFStudio, with room for other representations later.

Keep the existing runtime neutral. Tensor operators live in domains and
providers. Optional gsplat bindings live in integrations. A separate `studio`
application owns data, trainable representations, method recipes, optimizer
lifecycle, checkpoints, evaluation, exporters and the viewer. Methods,
renderers, parsers and exporters have independent registries and plugin entry
points. Adding a method must not add a method enum to core or an if-else ladder
to the trainer.

RaDe-GS uses Mollylulu/gsplat_radegs commit
`23b0f82551a938a05b5b99df06edf220b9305a51` via its public `RadeGS=True` API.
This dependency is opt-in and is never installed over an existing environment
automatically. Missing capability is an error, not a renderer substitution.
The paper-derived Torch renderer is explicitly selected for reference runs.

No complete permissive gsplat implementation of AmbiSuR or GOF was found in
the bounded public repository search. Their official repositories retain
restricted 3DGS/PGSR licenses. Implement their published mathematics here;
do not incorporate their source. Precomputed geometry priors are explicit
inputs. A model card's license on released meshes does not license source
code or third-party depth-estimator weights.

The initial workflow includes method configurations, COLMAP and transforms
parsers, image/mask/prior data management, train/resume, validation, mesh and
Gaussian export, geometry metrics and a local viewer. This is workflow and
extension coverage, not a claim of SDFStudio benchmark parity, all of its
methods, or validated paper-level reconstruction quality. Experimental
operator references have explicit semantic contracts and independent
analytic checks; they do not promote themselves to trusted release status.
