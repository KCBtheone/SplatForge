# Contributing

SplatForge uses the Developer Certificate of Origin (DCO) 1.1. By adding a
`Signed-off-by` trailer to a commit, a contributor certifies that the
contribution satisfies the DCO terms at <https://developercertificate.org/>.

Example:

```text
Signed-off-by: Contributor Name <contributor@example.com>
```

Operator work is intentionally split across protected changes:

1. specification, reference, fixtures, tests, tolerances, and provenance;
2. a maintainer-frozen work order;
3. an implementation-only provider change;
4. independent verification and trusted evidence.

An implementation change must not edit its own exam, thresholds, baseline,
work order, policy, or CI. A fixed provider must reject unsupported calls and
must never silently cast, copy, move, detach, synchronize, or fall back.

