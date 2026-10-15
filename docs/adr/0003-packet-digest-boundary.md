# ADR 0003: Work orders pin packet payloads without hashing themselves

- Status: accepted for the Phase 0 skeleton
- Date: 2026-09-12

## Decision

`WORK_ORDER.yaml.packet_digest` means the canonical digest of implementation
packet payload files excluding both `WORK_ORDER.yaml` and
`PACK_MANIFEST.json`. This avoids an impossible self-reference while still
letting the protected work order freeze every document, fixture, test,
benchmark, environment lock, and template exposed to the implementation agent.

`PACK_MANIFEST.json` separately records:

- the actual byte digest of `WORK_ORDER.yaml`;
- the sorted path, size, and digest of every packet file;
- the payload digest pinned by the work order;
- an aggregate digest over the manifest's semantic inventory.

Trusted dispatch additionally records the work-order digest and packet-manifest
digest outside the packet. Recomputing a manifest after tampering cannot change
the payload digest already pinned by the protected work order.

