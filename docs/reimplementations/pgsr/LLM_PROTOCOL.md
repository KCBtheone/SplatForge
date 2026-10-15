# LLM implementation protocol

Status: operational design. It becomes enforceable only after Phase 0 tooling and protected branches exist.

## Why one prompt is not enough

If one LLM is allowed to invent the contract, write the reference, implement the optimized provider and weaken the tests, a green test run has almost no evidentiary value. The process must separate the exam, the answer and the grading.

Use at least three isolated sessions plus the human maintainer:

| Role/session | Inputs | Outputs | Cannot change |
|---|---|---|---|
| Spec/reference author | public mathematical sources, architecture, source policy | OpSpec, readable reference, analytic fixtures, public tests, tolerance proposal | optimized provider |
| Implementation agent | frozen implementation packet only | provider patch, private debug tests, raw self-reports, known limitations | spec, reference, conformance, tolerance, baseline, CI |
| Verification agent | frozen contract plus candidate patch and independent test charter | adversarial/property tests, review, trusted evidence request | implementation and thresholds |
| Human maintainer | all evidence | freeze, exception, promotion and release decisions | cannot delegate accountability to an LLM statement |

For stronger evidence, split the first row into separate spec-owner and reference/test-author sessions.

The implementation session must be truly fresh: do not fork this conversation history, do not mount a PGSR or rasterizer checkout, and do not expose package/extension caches containing denied source. A general LLM may still have unknown pretraining exposure, so the process controls observable sources but does not claim perfect clean-room status.

## Before dispatch

The maintainer must complete all of the following:

1. Phase 0 runtime, manifest schemas, conformance harness and path gate work.
2. Merge the OpSpec/reference/tests/provenance in a protected change.
3. Calibrate tolerances and a trusted hardware baseline in separate protected changes.
4. Fill every digest, exact version, base commit and threshold in the work order.
5. Build the implementation packet and verify its aggregate digest.
6. Create an isolated branch/worktree from the exact base commit.
7. Remove credentials and denied source/cache mounts; disable network.
8. Apply filesystem permissions so only `allowed_paths` and self-report paths are writable.
9. Start a new LLM context with only the handoff prompt and packet.

Any `TBD`, mutable `main/latest` URL, missing digest or missing baseline makes the work order non-executable.

## First capability task: `matching.patch_zncc@1`

The first task should implement only the Triton provider for a frozen masked ZNCC contract. Patch warping is explicitly out of scope.

The intended test profile is:

- public semantics permit arbitrary sample count `S` in the Torch reference;
- the first Triton implementation declares only `S=49`;
- fp16 and fp32 inputs accumulate in fp32;
- mask and invalidity behavior are explicit;
- forward and first-order VJP are required;
- a fixed Triton provider rejects unsupported device/dtype/layout/sample-count cells;
- auto-policy fallback is tested elsewhere in the runtime;
- no dependency, API or test changes are allowed.

Suggested scoring:

| Area | Weight | Failure examples |
|---|---:|---|
| Numerical/boundary/gradient correctness | 40% | wrong masked mean, nonzero invalid gradient, threshold instability |
| Architecture and scope discipline | 20% | changes core, combines warp with ZNCC, edits frozen tests |
| GPU safety | 15% | out-of-bounds tail, wrong stream, hidden synchronization |
| Performance and memory | 15% | slower than frozen gate, material hidden copies, excessive scratch |
| Documentation/provenance | 10% | missing limitations, unlisted source, missing evidence |

Automatic failure overrides the numerical score.

## Automatic failure conditions

- Any path outside the work-order allowlist changes.
- A spec, reference, test, tolerance, baseline, work order, CI file or policy is changed.
- A test is deleted, skipped, xfailed or weakened.
- A fixed provider silently falls back, casts, copies, detaches or changes layout.
- Forward is correct but a required backward cell is missing or incorrect.
- Unsupported second-order differentiation silently returns a result.
- Evidence says only that compilation succeeded.
- The benchmark does not pin exact implementation identity or retain raw samples.
- Sanitizer reports an error, the kernel hangs/crashes, or a determinism claim fails.
- A dependency or source outside the allowlist is used.
- The agent accesses PGSR, a fork/mirror, a target rasterizer, a reimplementation or source-derived summary.

Denied-source access invalidates the session and worktree. Start again from the frozen base in a new context; do not ask the contaminated agent to rewrite the same patch.

## Evidence the implementation agent must return

The patch is not the whole delivery. Require:

```text
self_reports/<task-id>/
  environment.self_reported.json
  context_inventory.json
  tool_access_log.json
  conformance.self_reported.json
  properties.self_reported.json
  gradients.self_reported.json
  compile.self_reported.json
  sanitizer.self_reported.json
  benchmark.self_reported.json
  known_limitations.md
  CONTRACT_CHANGE_REQUEST.md       # only when needed
```

Self-reports are diagnostic, not release authority. Trusted CI reruns all checks from the prospective merge commit and emits formal evidence bound to the implementation artifact, spec revision, environment and exact provider ID.

## Verification session

The verifier should add tests derived from invariants, not from the candidate's implementation structure. For ZNCC, prioritize:

- positive and negative affine invariance away from thresholds;
- symmetry;
- masked-value and masked-gradient irrelevance;
- valid/invalid mixtures in one launch;
- near-constant values on both sides of the variance threshold;
- extreme but finite magnitudes;
- batch-vs-item equivalence;
- arbitrary upstream gradients and stride cases declared supported;
- empty, tail and very large batch counts;
- provider rejection without fallback;
- current-stream behavior and lack of host synchronization.

If the verifier finds a contract ambiguity, it files a contract-change request. It must not edit the provider or loosen a tolerance in the verification change.

## Promotion ladder

Use explicit claims:

1. `contract drafted`
2. `reference verified`
3. `provider correctness passed`
4. `provider GPU safety passed`
5. `provider performance gate passed`
6. `operator candidate`
7. `PGSR-style synthetic integration passed`
8. `PGSR-style training run passed`
9. `mesh workload validated`

Do not jump from a green ZNCC kernel to “PGSR implemented.” Rasterization backward, recipe integration, convergence and mesh reconstruction are independent work.

## Subsequent work orders

After the first task passes:

1. `sampling.patch_warp@1` Triton provider;
2. `camera.plane_induced_homography@1` and `spatial.depth_to_normal_cross@1` optimized cells only if profiling justifies them;
3. shortest-axis plane and planar-depth reference semantics;
4. `splat.rasterize_features@1` native CUDA forward;
5. the same fixed contract's backward;
6. a PGSR-style recipe/adapter using only public SplatForge APIs;
7. permitted real-scene and mesh validation.

The rasterizer task must be split into projection, bin/sort/coverage/composite semantics and fixed output behavior at the public boundary. Internal micro-kernels remain provider-private; do not expose the file structure of a paper repository as the public operator catalog.
