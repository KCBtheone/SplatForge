# ADR 0002: Provider changes are checked against the base work order

- Status: accepted for the Phase 0 skeleton
- Date: 2026-09-12

## Decision

The implementation scope checker reads the work order with `git show` from an
explicit protected policy ref. It does not trust a copy changed by the
candidate branch. The work order's `base_commit` identifies the frozen contract
tree from which the candidate diff is calculated; keeping the work order on a
separate protected ref avoids an impossible self-referential commit hash.

The checker verifies that `base_commit` is an ancestor of the separately named
candidate head, rejects every candidate-changed path outside the allowlist or
inside the denylist, rejects symlink traversal, and re-hashes all frozen inputs
from the checked-out validation tree. Trusted CI should use prospective merge
tree while passing the actual PR head as `--candidate-head`.

The protected policy ref also supplies
`tools/quality/protected_paths.yaml`. Those global protected patterns override
the work-order allowlist, so an overly broad or malformed work order cannot
authorize an implementation agent to edit the exam or the gate.

A work order with placeholders or a status other than `frozen` is not
executable. CI must execute the checker and policy from a protected base branch;
the script cannot make branch protection trustworthy by itself.

## Consequences

The current PGSR ZNCC work order remains `draft_not_executable`. It cannot pass
this gate until the repository has a trusted Git base, every input digest is
fixed, and the independent contract plus RTX 4080 SUPER evidence have been reviewed.
