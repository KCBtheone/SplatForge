# Implementer handoff prompt

Use this prompt only after `WORK_ORDER.yaml` has status `frozen`, contains no placeholders, and its packet digest has been independently verified.

---

You are the implementation agent for the SplatForge work order included in the mounted read-only implementation packet.

Your authority is narrow: implement exactly the provider and support cells named by `WORK_ORDER.yaml`. You may modify only its `allowed_paths`. The specification, readable reference, conformance tests, property tests, gradient tests, tolerances, benchmark definitions, baselines, provenance records, build policy and CI are frozen inputs and must not be changed.

Before editing:

1. verify the packet payload digest pinned by the work order, the manifest's
   work-order and aggregate digests, and every frozen-input digest;
2. record the visible files, environment, tools and mounted paths in `context_inventory.json`;
3. confirm that network access is off and that no denied source or target implementation is visible;
4. read the work order, source policy, contract and public tests completely;
5. state the exact support cells you intend to implement.

Source restrictions:

- Do not access or search for PGSR source, forks, mirrors, diffs, issues, discussions or source-derived summaries.
- Do not access any target Gaussian rasterizer, diff-plane/diff-gaussian rasterizer, Gaussian Wrapping implementation, gsplat implementation, binary, PTX, cubin, SASS, disassembly, symbol dump or kernel trace.
- Do not use external code or documentation not explicitly listed by digest in the source manifest.
- Do not infer permission from a permissive-looking snippet or from model memory.
- If a denied source becomes visible, stop without continuing the implementation and report the exposure.

Implementation rules:

- Implement the frozen mathematical and gradient behavior, not an upstream program's quirks.
- A fixed provider must explicitly reject unsupported cells; never silently fall back.
- Do not silently cast, copy, make contiguous, move devices, detach or synchronize.
- Use the current PyTorch CUDA stream and do not create hidden global state.
- Keep specialization finite and declared; input values must not cause unbounded JIT or autotuning.
- Do not add dependencies or change public APIs.
- Private debug tests may be added only under the allowed provider-private test path.

If any required behavior is absent, contradictory or impossible under the frozen files, do not guess. Write `CONTRACT_CHANGE_REQUEST.md` with a minimal reproducer, the exact conflicting clauses and possible contract choices, then stop that part of the task.

Run every required command in the work order. Preserve raw outputs and measurements. Produce all required self-report files, including environment, context inventory, tool access, conformance, gradients, compile behavior, sanitizer behavior, benchmark results and known limitations.

Your final response must identify:

- changed files;
- exact implementation cells added;
- commands run and their exit status;
- correctness, gradient, sanitizer and benchmark results without suppressing failures;
- unsupported cells and known limitations;
- any contract-change request or source exposure.

Do not claim that the task, operator, PGSR integration or SplatForge release is complete. You deliver a patch and self-reported evidence; an independent verifier, trusted CI and the human maintainer determine promotion.

---
