# PGSR and Unbiased Surfel in Studio

PGSR trains planar Gaussians with multiview regularization. Unbiased Surfel
trains surfels with corrected depth and depth convergence. Both fuse rendered
depths with Open3D TSDF and write `mesh.ply` in the run directory.

Install CUDA-enabled PyTorch and a CUDA toolchain. From the prepared source
root, install the Studio extras and the gsplat revision used by both methods.

```bash
python -m pip install '.[studio,mesh]' ninja
python -m pip install --no-build-isolation 'gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@6f378361f32ba4f1be86f65ae0db4948efb37dd5'
```

Studio reuses an importable `unbiased_gsplat_cuda` extension that provides
the required operators. Otherwise it builds the bundled Unbiased Surfel
kernels on first use. Set `TORCH_EXTENSIONS_DIR` to choose the build cache.
The two recipes can share this environment with GOF and VCR-GauS. Install the
optional RaDe-GS gsplat fork in a separate environment.

Provide COLMAP binary camera, image and point tables under `sparse/0/`
or `sparse/`, RGB images under `images/` or `images_1/`, and foreground
PNG masks under `masks/` or `masks_1/`. Mask basenames must match the RGB
images. PGSR also checks whether the capture is upright.

```bash
splatforge methods
splatforge config pgsr --output pgsr.yaml --data /path/to/colmap-scene
sf-train --config pgsr.yaml --output outputs/pgsr
sf-train unbiased-surfel --data /path/to/colmap-scene --output outputs/unbiased
sf-train --output outputs/unbiased --resume latest
sf-export --run outputs/pgsr --kind mesh
```

Each run saves `config.yaml` and a native `checkpoint.pt`. Use
`--set data.downscale=2` to downscale input images and `--device cuda:N`
to choose a visible GPU. Use `--set train.export_on_finish=false` to train
first and export later. `sf-export` reads the checkpoint directly and accepts
`--set export.voxel_size=...` and `--set export.truncation=...` in normalized
COLMAP coordinates. Resume PGSR with its original training settings. The
Unbiased Surfel Studio preset uses all input views.

The source preview retains the standalone `python pgsr/train.py` command.
Studio uses the native sources shipped with the wheel. In a development
checkout, `data.source_root` or `PGSR_SOURCE_ROOT` can select a local copy of
the shared sources.
