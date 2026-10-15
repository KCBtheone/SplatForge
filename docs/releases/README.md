# Preparing a public preview

The development workspace can contain private experiment outputs. Prepare a
new snapshot with the explicit source allowlist:

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
with that environment's Python. Run the companion's own tests separately.

Review the prepared snapshot before publishing. It contains selected source
from the working tree, including uncommitted implementations; it does not
copy the development repository's Git history or local runtime artifacts.
The main source distribution excludes the independently installed companion;
the complete repository source ZIP includes it. A release ZIP should be made
from the reviewed Git commit rather than the workspace directory.

Do not reuse an existing snapshot directory. Retain the release tag, the
published packages, and the compact verification record for each preview.
