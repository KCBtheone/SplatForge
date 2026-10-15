# Contract freeze checklist

Status: required decisions before any PGSR-style implementation work order becomes executable.

The PGSR paper describes the mathematical direction but does not define a production operator contract. Every item below needs an explicit answer in an OpSpec, reference, test, manifest or policy. An implementation LLM is not allowed to fill these gaps by guessing.

## Global tensor contract

- Exact public schema and OpKey major version.
- Shape grammar, including batch prefixes and empty dimensions.
- Whether broadcasting is forbidden, limited or fully specified.
- Accepted dtypes and accumulation dtype per provider cell.
- Device combinations and cross-device rejection.
- Required strides/layout and whether copies are allowed.
- Output dtype, shape, layout, aliasing and mutation behavior.
- Finite-input requirement and NaN/Inf behavior.
- Determinism claim and its scope.
- Autograd inputs, first/second-order support, JVP/VJP support and nondifferentiable outputs.
- FakeTensor, `torch.compile`, export and autocast behavior.
- Structured failure type for unsupported cells; no silent fallback for a fixed provider.

## Camera and pixel conventions

- `world_to_camera` versus `camera_to_world` naming and exact equation.
- Column-vector versus row-vector multiplication.
- Right-handed/left-handed coordinate system and camera forward axis.
- Image tensor axis order and `(u,v)` meaning.
- Pixel centers: integer centers, half-pixel centers or another convention.
- Intrinsics definition, skew support and image resize/crop rules.
- Whether rays are `K^-1 [u,v,1]`, normalized unit rays or z-normalized rays.
- Whether returned depth is camera-z, ray parameter or Euclidean range.
- Validity for zero/negative z and points on the near plane.

## Shortest-axis plane extraction

- Scale parameterization: raw positive scale, log scale or activated scale.
- Quaternion order, normalization, handedness and matrix convention.
- Exact `argmin` tie-break.
- Gradient at and near scale ties.
- Behavior for zero, negative, non-finite or extremely small scales.
- Normal sign rule and exact viewing-vector direction.
- Whether sign selection is differentiable or detached.
- Behavior when the view vector is zero or the normal is perpendicular to it.
- Whether the plane is per-Gaussian/per-view or world-oriented with a later camera transform.

## Homogeneous plane pair and depth recovery

- Camera-space plane equation and sign convention.
- Transformation of normals under rigid and, if supported, non-rigid transforms.
- Whether normals must be unit length before offset construction.
- Preserve the coupled pair `(normal_numerator, distance_numerator)`; never normalize one alone.
- Denominator epsilon, comparison operator and scale dependence.
- Grazing-ray, zero-alpha, empty-pixel, behind-camera and negative-depth behavior.
- Invalid output value: zero, NaN, sentinel or unspecified tensor plus mask.
- Gradient on invalid branches and at threshold equality.
- Output and accumulation precision.

## Projected Gaussian and ellipse semantics

- Covariance parameterization and positive-definiteness requirements.
- Projection Jacobian and camera clipping behavior.
- Minimum screen-space variance or antialias term.
- Footprint cutoff and exact ellipse coverage formula.
- Image-bound convention and integer tile coverage.
- Sort key, stable tie-break and treatment of non-finite depth.
- Opacity activation, alpha clamp and low-alpha cutoff.
- Front-to-back transmittance formula and accumulation order.
- Early termination threshold and whether it is semantic or provider-private approximation.
- Background/empty output values.
- Payload channel rules and accumulation dtype.
- Gradient scope for ellipse, depth, opacity and payload.
- Whether sorting/binning is differentiated.
- Upper bounds, index width, scratch ownership and stream behavior.

## Depth-to-normal

- Neighbor offsets and whether they are one or more pixels away.
- Cross-product operand order and normal orientation.
- Boundary policy and validity propagation from depth.
- Unprojection convention and depth kind.
- Zero/near-zero cross product threshold.
- Normalization epsilon and output for invalid normals.
- Edge-aware image gradient definition, channel reduction and normalization if used by a recipe.

## Plane-induced homography

- Direction and naming of relative pose.
- Coordinate system in which `normal` and `distance` live.
- Matrix multiplication and homogeneous point convention.
- Behavior for `distance` near zero.
- Homogeneous normalization epsilon and sign.
- Whether points are normalized after each leg of a roundtrip or only at the end.
- Pixel-space error units and norm.
- In-bounds and positive-depth validity in both views.
- Equality behavior for the one-pixel occlusion threshold if following the paper profile.
- Which weight is detached and whether the validity mask is differentiable.
- Denominator used for the final reduction: all pixels, valid pixels or weighted count.

## Patch warp

- Input layout and channel order.
- Patch coordinates and sample ordering.
- Bilinear interpolation formula.
- Padding/out-of-bounds policy and output mask.
- Pixel-to-normalized-grid mapping and `align_corners` equivalent.
- Whether a sample partially outside is valid.
- Image gradient and grid gradient at integer positions and boundaries.
- NaN/Inf coordinate behavior.
- Accumulation/output dtype and autocast.
- Whether a fixed 7 by 7 specialization is a provider cell rather than the public API.

## Patch ZNCC

Recommended v1 semantic shape, still requiring an independent freeze PR:

```python
score, valid = sf.matching.patch_zncc(
    lhs,             # [..., S]
    rhs,             # [..., S]
    sample_mask,     # bool [..., S]
    variance_epsilon,
)
```

Decisions to freeze:

- `patch_zncc` consumes already sampled scalar patches; grayscale conversion and warp are separate.
- `lhs`, `rhs` and mask have identical shapes; no implicit broadcasting.
- Last dimension is sample dimension; `S=49` is a provider specialization, not core semantics.
- Masked means and centered sums versus normalized variances.
- Minimum valid sample count.
- Exact variance threshold, units, dtype and strict/non-strict comparison.
- Invalid score and input gradients.
- Whether result is clamped to `[-1,1]`; recommended v1 is no clamp.
- Output dtype; recommended fp16/fp32 input cells use fp32 accumulation and score.
- Input finite requirement and output finite guarantee.
- Determinism and reduction order claims.
- First-order gradient formula and second-order behavior.
- Empty batch and zero-length sample behavior.
- Fixed Triton provider support: CUDA, `S=49`, last dimension contiguous; fixed-provider mismatch must reject.

Analytic fixtures must include identical, positive-affine, negative-affine, orthogonal zero-mean, constant, near-threshold, partial-mask, single-sample, no-sample and mixed-validity batches. Properties must cover symmetry, masked-value irrelevance, affine invariance away from the validity threshold, batch equivalence and finite outputs.

## Recipe-only decisions

These are versioned algorithm choices but should not become runtime/provider policy:

- flattening-loss norm and weight;
- edge-aware loss formula and gradient normalization;
- 7 by 7 grayscale PGSR profile;
- neighbor graph thresholds and random selection;
- multi-view loss weights and start schedule;
- hard gates such as reprojection error below one pixel;
- exposure-compensation gate and loss mix;
- depth ray-normal filter threshold;
- densification/pruning strategy;
- TSDF and mesh extraction configuration.

## Verification freeze

- Hand-derived fixtures are committed before optimized implementation begins.
- Reference uses clear Torch composition and supports CPU float64 where meaningful.
- Public conformance, property and gradient tests have immutable digests.
- Hidden adversarial tests are authored by an independent verifier and are disclosed after the evaluation cycle when practical.
- Tolerances are calibrated in a separate protected change and never loosened in the implementation PR.
- Benchmark shapes, exact implementation identity, warmup, repetitions, timing method and memory method are frozen.
- Hardware/software environment and performance baseline are trusted artifacts.
- Sanitizer checks cover tails, empty input, large indexing and all declared layouts.

## Provenance freeze

- Every allowed source has exact URL/version/content digest and permitted role.
- Unlisted sources are denied.
- Work-order base commit and packet digest are exact.
- Actor context inventory and tool access log are required.
- The implementation agent uses a fresh context and an offline workspace with no denied checkout/cache.
- Any denied-source access invalidates the worktree/session; deleting browser history does not repair provenance.
- The release claim stays within source-controlled independent implementation.
- Patent status remains explicitly `unreviewed` until separate review.
