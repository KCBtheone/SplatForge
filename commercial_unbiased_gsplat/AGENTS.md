# Commercial Unbiased Surfel source boundary

These instructions apply to every file under this directory.

- Do not open, search, summarize, execute, import, link, copy, translate, or
  inspect source or binaries from `XiaoXinyyx/Unbiased_Surfel`, its
  `diff-surfel-rasterization` submodule, or the local isolated reference
  checkout named in the original work order.
- Apply the same source-isolation boundary to the PGSR implementation: its
  public paper/supplement and repository license text may be read, but its
  Python/CUDA source, binaries, traces, checkpoints, patches, and derived
  implementation notes must not be opened or used in this project.
- Implementation sources are limited to the public paper/supplement, official
  gsplat at commit `6f378361f32ba4f1be86f65ae0db4948efb37dd5`, this
  directory's specifications, and source-free input/output tensor fixtures.
- Track R fixtures must contain data and neutral metadata only. Quarantine a
  fixture before opening it if it contains code, patches, binaries, traces,
  implementation pseudocode, or author-internal symbols.
- Keep mathematical operators out of trainer code. Change tensor semantics only
  by updating `SPEC.md`, focused tests, and provenance together.
- Do not enable packed mode by guessing index mapping. Require a real pinned
  gsplat probe and dedicated tests.
- Run `PYTHONPATH=. python -m unittest discover -s tests -v` after changes.
- Preserve evidence boundaries in `STATUS.md`: synthetic, pinned-GPU,
  source-free reference, training, performance, and license closure are
  separate gates.
- Never describe this engineering isolation process as legal or patent
  clearance.
