# Roadmap

SplatForge aims to become a common toolbox for reconstruction research:
reusable operators, a shared experimental framework, and method recipes that
help researchers develop and evaluate new ideas.

Geometry-oriented Gaussian Splatting is the starting point. The broader
direction includes additional GS methods, selected multi-view stereo
components, and geometry processing. The stages below describe intended work,
not capabilities already delivered.

## Phase 0 — test the abstractions

**Current stage: prototype and feasibility work.**

Implement selected methods and operators to investigate their interface,
gradient, and execution requirements. The public code currently includes
RaDe-GS, AmbiSuR, and GOF Studio recipes, plus an independent Unbiased Surfel /
PGSR-style auxiliary package.

Questions for this stage include:

- Which components can be expressed with useful tensor contracts?
- Which components depend on a particular representation or renderer?
- Can operators be reused without introducing hidden algorithm changes?
- Can shared training, checkpoint, export, and evaluation infrastructure make
  experiments easier to inspect and compare?

The implementations and tests are evidence for answering these questions.
Completing a synthetic workflow does not close method reproduction or
real-scene quality validation.

## Phase 1 — build a usable GS workbench

**Planned: a framework and the first coherent operator library.**

- A documented set of geometry and GS operators, including tensor conventions,
  gradients, reference behavior, and supported execution configurations.
- A shared workflow for loading scenes, training, resuming, evaluating,
  visualizing, and exporting geometry.
- Examples showing how to add a method, reuse an operator, or replace a
  component within a supported recipe.
- Numerical and performance validation for accelerated providers as they are
  introduced.
- Method-specific reproduction records that distinguish implementation,
  completed experiments, quality, and performance.

The intended outcome is that a researcher can spend more time on the component
they want to change and reuse the surrounding infrastructure.

## Phase 2 — extend the reconstruction toolbox

**Longer-term direction: more methods and neighboring capabilities.**

Explore additional GS representations and methods, selected MVS components
such as multi-view matching and depth consistency, and geometry utilities for
point clouds, fusion, and meshes. Extend common data conventions and component
interfaces where the earlier experiments show useful boundaries.

The ambition is to connect these capabilities into reusable research workflows.
Their scope and interfaces will depend on what proves useful in the GS work.
There is no fixed delivery schedule for this stage.
