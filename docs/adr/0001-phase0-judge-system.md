# ADR 0001: Phase 0 judge-system boundary

- Status: accepted for the Phase 0 skeleton
- Date: 2026-09-12

## Decision

SplatForge separates an immutable operator definition from each implementation
manifest. Runtime dispatch constructs a call signature from tensor metadata and
static attributes, filters candidates using policy and capability evidence, and
binds an exact implementation in an execution plan. Runtime failure is terminal;
the plan never retries another provider.

Phase 0 uses typed Python validators rather than claiming that the illustrative
shape notation in `ARCHITECTURE.md` is already a complete DSL. The registries
are independently instantiable, reject replacement, increment generations only
after successful transactional additions, and may be explicitly sealed.

A capability record is separate from the provider-authored declaration. An
untested or experimental implementation may be selected only by exact ID with
`allow_unverified=True`; quarantine and disabled states cannot be overridden.

## Consequences

The virtual `testing.scale_shift@1` operator can prove the lifecycle without
introducing PGSR or Gaussian semantics. It is not a release API. Compile,
FakeTensor, export, signed catalogs, and artifact-specific cell predicates are
not claimed by this first skeleton and require later dedicated vertical slices.

