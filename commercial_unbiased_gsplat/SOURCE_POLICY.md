# Source policy

## Allowed inputs

- public paper and supplementary material;
- the public PGSR paper/supplement and its repository license text solely for
  attribution and source-policy classification;
- official gsplat source/doc/license at commit
  `6f378361f32ba4f1be86f65ae0db4948efb37dd5`;
- this repository's own specifications, tests, and generated evidence;
- source-free Track R input/output tensors with provenance metadata.

## Prohibited inputs and runtime links

- all source and binaries from `XiaoXinyyx/Unbiased_Surfel` and its
  `diff-surfel-rasterization` submodule;
- PGSR implementation source, binaries, checkpoints, traces, patches,
  source-derived pseudocode, and runtime dependencies;
- source-derived pseudocode, patches, traces, symbol-by-symbol descriptions, or
  translations from those materials;
- runtime imports, dynamic loading, subprocess calls, or linkage to them.

Reference fixtures are data-only. A fixture that contains prohibited material
must be quarantined and must not be opened in the commercial implementation
context.
