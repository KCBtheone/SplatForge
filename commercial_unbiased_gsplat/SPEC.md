# Tensor contract: source-isolated Unbiased Surfel adapter v1

This document is normative for the first pure-PyTorch implementation. It fixes
engineering choices that the public equations or gsplat interface expose and
marks choices that still require source-free reference fixtures.

## Pinned rasterizer contract

- gsplat identity: `6f378361f32ba4f1be86f65ae0db4948efb37dd5`;
- entry point: `rasterization_2dgs(..., packed=False)`;
- accepted hit list: `rasterize_to_indices_in_range_2dgs(0, 2**31-1, ...)`;
- hit order: contiguous dense-pixel segments, within each segment the existing
  gsplat tile list order (Gaussian center z-depth sort);
- hit membership remains owned by gsplat: exact ray-cross rejection, Gaussian
  evaluation, `alpha < 1/255` cutoff, and exclusive `next_T <= 1e-4` stop;
- this adapter raises on non-contiguous segment output and never silently sorts,
  adds, or removes hits.

Packed metadata is outside v1. Its camera/Gaussian remapping is recorded by the
probe but is not inferred in production.

## Per-hit schema and equations

Inputs after gathering are:

| Tensor | Shape | Meaning |
|---|---:|---|
| `ray_transforms` | `[M,3,3]` | gsplat rows `u_M, v_M, w_M` |
| `means2d` | `[M,2]` | projected center in pixel coordinates |
| `pixel_ids` | `[M]` | row-major pixel id inside one image |
| `opacities` | `[M]` | gsplat per-view opacity, not opacity times G |
| `segment_ids` | `[M]` | `image_id*H*W + pixel_id`, nondecreasing |

At pixel center `p=(x+0.5,y+0.5)`:

```text
h_u = p_x*w_M - u_M
h_v = p_y*w_M - v_M
h = cross(h_u,h_v)
s = h.xy/h.z
rho3 = s_x^2+s_y^2
rho2 = 2*||means2d-p||^2
rho = min(rho3,rho2)
G = exp(-rho/2)
d_ray = s_x*w_M.x+s_y*w_M.y+w_M.z
d = d_ray if rho3<=rho2 else w_M.z
```

`h.z == 0` and non-finite intermediates are invalid, matching the pinned CUDA
hit kernel's exact-zero rejection. `hz_epsilon` defaults to zero. Setting it
above zero is an explicit experiment change and must be recorded. Invalid
numeric outputs store zero only as a carrier; the validity mask is authoritative
and zero must never be interpreted as recovered depth.

## Convergence

Only adjacent entries in one unchanged segment form a pair. A pair is valid
when both hits are valid and `abs(d_i-d_(i-1)) <= scene_radius/4`.

```text
pair_i = stop_gradient(min(G_i,G_(i-1))) * (d_i-d_(i-1))^2
```

The custom first-order VJP is:

```text
d pair / d d_(i-1) = -2*w*(d_i-d_(i-1))
d pair / d d_i     =  2*1.25*w*(d_i-d_(i-1))
```

Opacity is not an input. Reduction is mandatory and its observed denominator is
returned. The v1 trainer default was `mean_segments`, whose denominator is the
fixed dense image-pixel count, not the variable hit/pair count.

v2 experimental trainer default is `mean_active_segments` (background pixels
with no valid pairs excluded). Rationale: on Object1 the `mean_segments`
convergence term measured `~0.0005` when converged (weighted `~0.003` at paper
weight 7.0 against color loss `~0.1`), i.e. geometrically negligible —
baseline-vs-joint corrected masks agreed at IoU `0.9998`. The v2 defaults pair
`mean_active_segments` with experimental weights (`convergence 15.0`, `normal
0.2`): at init (scattered, raw conv `~0.105`) the weighted term is `~1.6`,
decaying to `~0.015` when converged; normal lands at `~0.08 -> ~0.02`. Paper
constants (`7.0`/`0.05`) are retained in the table below for reference and
remain the tensor-library defaults; the v2 training defaults are explicitly
not a paper claim.

## Corrected depth

Within the unchanged hit order:

```text
contribution_i = (opacity_i + 0.1) * G_i
O_i = segmented_cumsum(contribution)_i
selected = first i where O_i > 0.6
```

The comparison is strict. No-hit and no-crossing segments are invalid; first-hit
crossing selects that hit. Local inversion of actual ray depths does not trigger
resorting. The selected depth is `d`, never gsplat center or median depth.

Accumulation currently uses the input dtype. The source-free Track R fixtures
must freeze threshold-near floating-point expectations before any alternative
accumulation dtype or fused CUDA path is promoted.

## Depth normal and normal loss

Corrected depth is camera z-depth. Pixel-center pinhole rays are backprojected
and transformed with `camera_to_world`. Central differences require valid
center, left, right, upper, and lower points. Image borders, non-finite points,
and cross products of magnitude at or below `normal_epsilon` are invalid.
Normals are oriented toward the camera and expressed in world coordinates.

The production adapter consumes gsplat's primary `render_normals` and
`render_alphas` outputs. The pinned projection metadata stores per-Gaussian
normals in camera coordinates, but a direct CUDA probe confirms that the
primary accumulated `render_normals` output is already in world coordinates.
The dense path therefore does not rotate it again. Since these values are
respectively `sum_i(weight_i*n_i)` and `sum_i(weight_i)`, it evaluates exactly:

```text
alpha_accum - dot(normal_accum,N)
  = sum_i weight_i * (1-dot(n_i,N))
```

This avoids a second approximation to gsplat transmittance. A hit-expanded
Torch version exists only for tensor tests and diagnostics.

## Published training constants

| Setting | Value |
|---|---:|
| D-SSIM color weight | `0.2` |
| normal weight | `0.05` |
| convergence weight | `7` |
| asymmetric far-depth gradient | `1.25` |
| densification stop | iteration `20_000` (exclusive helper boundary) |
| corrected opacity compensation | `0.1` |
| corrected threshold | `0.6` (strictly exceeded) |
| DTU-only gamma | `0.5` |

The post-densification scale limit is not guessed. The host trainer must pass
its actual densification scale threshold to the explicit tangent-scale clamp
hook.

## Paper-derived PGSR-style multi-view adapter

This optional adapter is independently expressed from the public PGSR paper
and supplement. It does not use the PGSR implementation. The project calls the
mode `joint_pgsr_mv` and deliberately describes it as **PGSR-style**, not as a
code-equivalent PGSR reproduction.

### Camera graph

The graph is built over the host trainer's normalized training cameras. Camera
forward direction is the camera-to-world positive z axis. For each reference
camera, candidate neighbors must satisfy both:

```text
relative optical-axis angle <= 30 degrees
0.01 <= Euclidean camera-center distance <= 1.5
```

Candidates are sorted lexicographically by angle, distance, source-image id,
and dataset-item id, then limited to eight. One graph neighbor is selected by
a seeded uniform random draw on each training iteration. Graph entries are
training-dataset item ids; the manifest separately records source-image ids so
split indexing cannot be confused with COLMAP indexing.

### Plane homography and geometric consistency

For source-to-target camera motion
`inverse(T_target_camera_to_world) * T_source_camera_to_world = [R|t]`, a
source-camera plane uses `n^T X + d = 0`, where `d=-n^T X_point`. Its homography
is:

```text
H_target,source = K_target * (R - t*n^T/d) * inverse(K_source)
```

`X_point` is the world point backprojected from this project's corrected
ray/surfel depth. `n` comes from the normalized primary gsplat accumulated
world-normal direction. These two inputs are an explicit integration choice:
they join the published multi-view constraint to the Unbiased Surfel output
without substituting center or median depth.

Pixel coordinates denote centers `(x+0.5,y+0.5)`. Bilinear sampling uses
`align_corners=False`. The forward homography maps a reference pixel to the
neighbor. The neighbor point and normal define a reverse homography back to
the reference, giving:

```text
phi = ||p_reverse - p_reference||_2
valid_geometry = finite and phi < 1 pixel
omega = stop_gradient(exp(-phi))
L_geometry = sum(valid_geometry * omega * phi) / count(valid_geometry)
```

The reference and bilinearly sampled neighbor corrected-depth/normal pixels
must both be valid and inside the SAM2 foreground masks. Empty reductions
return differentiable zero with an explicit zero denominator.

### Patch photometric consistency

At geometry-valid samples, the forward homography warps a 7x7 reference
grayscale patch into the neighbor. Zero-mean normalized cross-correlation is:

```text
NCC = dot(I_ref-mean(I_ref), I_nei-mean(I_nei))
      / sqrt(sum((I_ref-mean(I_ref))^2) * sum((I_nei-mean(I_nei))^2))
L_photo = sum(valid_photo * omega * (1-NCC)) / count(valid_photo)
```

Both complete patches must be in bounds and inside their SAM2 masks. Patches
whose centered energy is at or below `1e-6` are invalid. To keep the pure-Torch
adapter tractable at 720x1280, patch centers use a recorded stride of four
pixels. This stride is a project performance choice rather than a claim about
the paper implementation.

The joint training objective adds the published PGSR weights to this project's
existing terms:

```text
L = L_color + 7*L_convergence + 0.05*L_normal
    + 0.03*L_geometry + 0.15*L_photo
```

The default retains gradients through both renders. A CLI-only
`mv_stop_gradient_neighbor` fallback exists for a measured runtime OOM; if
used, it must be recorded in the run manifest and reported as a one-sided
variant rather than silently treated as the default method. v2 experimental
training defaults are additionally: photometric weight `0.05` (down from the
paper `0.15`, which dominated at `~0.06` vs color `~0.1` while NCC showed
`~0.4` misalignment), and `mv_start_iter 5000` so coarse geometry establishes
before pairwise terms engage. In that one-sided
path the neighbor computes the same ordered-hit ray depth, corrected-opacity
threshold, world-space backprojection, and normalized primary-render normal as
the full adapter. It intentionally skips neighbor convergence and depth-normal
loss construction. The accepted hit list is scanned in contiguous chunks of at
most 1,048,576 hits. A carry is retained only when one dense pixel segment
crosses a chunk boundary; the first strict threshold crossing is still selected
in the original hit order. Chunk intermediates are released before the next
chunk, and all remaining per-hit state is released before pairwise consistency.
Only dense neighbor points, normals, and validity masks remain live. The chunk
size is recorded in the run manifest. This is a bounded-lifetime optimization,
not a change to the multi-view equations, image resolution, or hit order.

### 2DGS+PGSR training mode

The `2dgs_pgsr_mv` training mode runs the same pairwise consistency math
without any Unbiased Surfel auxiliary: the reference and neighbor surfaces
are official gsplat expected depth backprojected to world points with the
normalized primary accumulated world normals. Expected depth and accumulated
normals are both differentiable rasterizer outputs. Validity is finite
expected depth with accumulated alpha above 0.01, the same five-pixel stencil
as the corrected path, and the SAM2 foreground mask. Period, start
iteration, weight scaling, and one-sided neighbor provenance are recorded in
the run manifest exactly as in the joint mode.

## Evidence not supplied by this contract

- agreement with the non-commercial author implementation;
- reference-gradient tolerances;
- packed-mode equivalence;
- dataset convergence or final geometry quality;
- an optimization decision in favor of fused CUDA;
- transitive license or patent clearance.
- equivalence to the PGSR repository implementation or its training traces;
- evidence that the stride-four pure-Torch patch sampling matches an
  unpublished implementation detail.
