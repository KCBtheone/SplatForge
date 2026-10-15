# Roadmap

SplatForge starts with geometry-oriented Gaussian Splatting and an operator
library at its core. The framework is a way to exercise and test those
components.

| Stage | Goal | Status |
| --- | --- | --- |
| Phase 0: feasibility | Implement and test selected methods and operators; explore interface boundaries and reuse | Current prototype |
| Phase 1: framework and operators | Establish a usable framework and a first set of well-tested geometry GS operators | Planned |
| Phase 2: broader reconstruction tooling | Explore additional GS methods, selected MVS components, and geometry processing | Longer-term direction |

The current preview does not declare Phase 0 complete. Components that work on
synthetic inputs still need method-specific real-data, numerical, and
performance validation. A stable API and free interchangeability between
arbitrary paper components are research goals.
