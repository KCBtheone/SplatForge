# Preparing a public preview

Prepare a public source tree with the release allowlist. It includes Studio,
the operator library, selected tests, and the companion package. The research
orchestration tools, internal notes, and experiment outputs are kept in the
development checkout.

The Studio wheel includes GOF, VCR-GauS, PGSR, and Unbiased Surfel.
It includes the CUDA sources for VCR-GauS and Unbiased Surfel. The full
source preview also includes the standalone operator package.

```bash
python tools/release/prepare_public.py --destination build/preview/source
cd build/preview/source
python -m pip install -e '.[studio,test]'
python tools/quality/run_phase0_checks.py
python -m build --outdir ../artifacts
```

`python -m build` builds the wheel from the generated source distribution.
Install that wheel in a separate environment containing the Studio dependencies
and run `tools/release/verify_installed.py --work <new-directory> --report <file>`
with that environment's Python. The installed-wheel check covers the registry
and bundled resources. See the [CUDA setup and input requirements](pgsr-preview.md)
for training PGSR and Unbiased Surfel.

Review and build from the prepared tree. It takes the current working files,
including uncommitted edits. Publish the reviewed public tree and build
artifacts from that tree.

Do not reuse an existing snapshot directory. Retain the release tag, the
published packages, and the compact verification record for each preview.
