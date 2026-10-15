# PGSR-style independent implementation pack

Status: planning pack, not yet a frozen implementation pack.

This directory answers one practical question: how should SplatForge ask a separate LLM to implement the operators needed by a PGSR-style reconstruction recipe without turning the repository into a fork of a paper codebase?

The answer is:

1. treat PGSR as a consumer recipe, not as the core ontology;
2. derive operator contracts from public mathematics and SplatForge's own requirements;
3. freeze the contract, reference, tests, tolerances, benchmark and source policy before implementation;
4. give the implementation LLM one bounded provider task in a fresh context;
5. verify its patch in an independent context and trusted CI;
6. only then increase the task size.

The first implementation task should be `matching.patch_zncc@1`, not a complete planar-Gaussian rasterizer. It is directly useful for the paper's multi-view photometric term, yet small enough to expose whether an LLM can correctly handle indexing, masks, low-variance inputs, first-order gradients and GPU performance.

## What is in this pack

- [OPERATOR_DAG.md](OPERATOR_DAG.md): a paper-derived decomposition into reusable operators, recipes and adapters.
- [SOURCE_POLICY.yaml](SOURCE_POLICY.yaml): the source allowlist, denylist, actor separation and current exposure record.
- [CONTRACT_FREEZE_CHECKLIST.md](CONTRACT_FREEZE_CHECKLIST.md): every observable behavior that must be decided before a work order can be frozen.
- [FREEZE_READINESS.md](FREEZE_READINESS.md): current Phase 0 results and the exact blockers that keep the first work order non-executable.
- [FIRST_TASK_CONTRACT_PROPOSAL.md](FIRST_TASK_CONTRACT_PROPOSAL.md): a concrete, reviewable starting contract and test plan for the ZNCC capability task.
- [SPEC_REVIEW_QUESTIONS.md](SPEC_REVIEW_QUESTIONS.md): semantic and execution conflicts that must be resolved before freezing that proposal.
- [LLM_PROTOCOL.md](LLM_PROTOCOL.md): the multi-session workflow, promotion ladder and scoring rules.
- [SPEC_AUTHOR_HANDOFF_PROMPT.md](SPEC_AUTHOR_HANDOFF_PROMPT.md): a fresh-context prompt for producing the independent contract/reference change.
- [IMPLEMENTER_HANDOFF_PROMPT.md](IMPLEMENTER_HANDOFF_PROMPT.md): the prompt to use after the packet has real hashes and a trusted base commit.
- [VERIFIER_HANDOFF_PROMPT.md](VERIFIER_HANDOFF_PROMPT.md): an independent adversarial review prompt.
- [../../../work_orders/drafts/SF-PGSR-001-ZNCC-TRITON.yaml](../../../work_orders/drafts/SF-PGSR-001-ZNCC-TRITON.yaml): the first machine-readable work-order draft.

## Hard boundary

The PGSR repository is not an implementation source for SplatForge. Its published software license limits use to educational, research and non-profit purposes and directs commercial users to contact the authors. The implementation agent therefore receives no PGSR source, fork, binary, disassembly, source-derived summary or target-kernel trace.

The official paper is a specification-author source, not an implementation-agent source. Do not commit its PDF, figures or copied prose into the Apache-2.0 repository. Record a versioned link and independently restate the mathematical contract using SplatForge's own notation.

This process should be described publicly as a **source-controlled independent implementation**. It is not proof of a legally certified clean-room implementation: a general-purpose LLM's pretraining exposure normally cannot be established.

## The two packs

### Spec pack

Only the spec owner gets:

- versioned PGSR paper links;
- the official project page for scope only;
- the official license for compliance review;
- this operator map and ambiguity checklist;
- a rule prohibiting navigation into repository code.

The spec owner produces original SplatForge contracts. A separate reference/test author should turn those contracts into a readable float64-capable Torch reference, analytic fixtures and public conformance tests.

### Implementation pack

The implementation LLM gets only frozen SplatForge-owned material:

```text
SF-PGSR-001/
  WORK_ORDER.yaml
  PACK_MANIFEST.json
  SOURCE_POLICY.yaml
  CONTEXT_POLICY.md
  op_spec/
  reference_readonly/
  fixtures_public/
  tests_public/
  benchmark/
  environment/
  provider_example_unrelated/
  evidence_schemas/
  CONTRACT_CHANGE_REQUEST.md
  SUBMISSION_CHECKLIST.md
```

The packet must be content-addressed. The work order freezes the digest of every protected input and the aggregate packet digest. The implementation environment is offline and only provider-private paths plus a self-report directory are writable.

## Recommended work sequence

| Stage | Work order | Why it comes here |
|---|---|---|
| 0 | Phase-0 registry, schemas and conformance harness | Creates a judge before asking an LLM to write an answer |
| 1 | `camera.project_pinhole@1` vertical slice | Proves the operator lifecycle and dispatch path |
| 2a | `matching.patch_zncc@1` Triton forward/backward | First bounded LLM capability test |
| 2b | `sampling.patch_warp@1` Triton provider | Adds interpolation, boundary and grid-gradient semantics |
| 3 | plane homography, depth-to-normal and planar-depth operators | Freezes camera and degeneracy conventions |
| 4a | generic `splat.rasterize_features@1` native CUDA forward | First real splat kernel, still representation-neutral at the raster boundary |
| 4b | the same operator's backward | Makes training possible |
| 5 | PGSR-style external recipe/adapter | Integrates operators without making runtime depend on PGSR |
| 6 | real workload validation | Separately measures training, depth, mesh quality, speed and memory |

Do not dispatch the draft work order before Stage 0 exists and every `TBD` field is replaced by trusted, immutable evidence.

## What passing the first task proves

Passing `SF-PGSR-001` proves only that the selected LLM can implement one frozen Triton provider under source and path constraints. It does not prove that:

- the LLM can write a safe differentiable rasterizer;
- a PGSR-style training recipe converges;
- mesh extraction is correct;
- the end-to-end reconstruction meets a customer-quality or time target.

Those are separate claims with separate evidence gates.

## Primary sources used for this planning pack

- PGSR, arXiv:2406.06521v2: https://arxiv.org/abs/2406.06521v2
- PGSR HTML paper, including supplementary material: https://arxiv.org/html/2406.06521v2
- Official project page: https://zju3dv.github.io/pgsr/
- Official software license: https://github.com/zju3dv/PGSR/blob/main/LICENSE.md

No target implementation source is normative for this pack.
