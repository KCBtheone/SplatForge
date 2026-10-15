# PGSR-style operator DAG

Status: paper-derived design input. This is not a frozen OpSpec.

The purpose of this decomposition is not to reproduce a repository file-for-file. It is to identify stable mathematical seams that remain useful for Gaussian, point, image-matching and later non-Gaussian consumers.

## End-to-end dataflow

```text
camera + Gaussian parameters
       |
       +--> camera projection/rays -----------------------------+
       |                                                        |
       +--> covariance decode + covariance projection           |
       |                     |                                  |
       |                     +--> screen ellipse + sort depth --+--+
       |                                                           |
       +--> shortest-axis plane --> camera normal + plane offset ---+|
                                                                    ||
                 payload = [color/features, camera normal, offset]  ||
                                                                    vv
                    ordered splat feature accumulation + alpha
                                      |
                    +-----------------+------------------+
                    |                                    |
             rendered image                  blended plane pair (n_bar, d_bar)
                                                         |
                                            ray/plane depth recovery
                                                         |
                    +----------------+--------------------+
                    |                |                    |
             depth-to-normal   plane homography     depth validity/filter
                    |                |
          single-view recipe   +----+-------------------------+
                               |                              |
                          patch warp --> patch ZNCC    forward/backward error
                               |                              |
                               +--------- multi-view recipe --+
                                                         |
                                               training recipe/adapter
                                                         |
                                              depth images --> TSDF/mesh
```

The thin waist is not a `PGSRModel` object. It is a set of tensor contracts. A point-cloud consumer may use projection and matching without importing Gaussian code; a mesh consumer may use camera rays and sampling without importing splat code.

## Paper locator and notation cautions

For the official v2 PDF, the relevant locations are:

- 3DGS projection and alpha compositing: PDF page 4;
- Eq. 1-2: PDF page 5;
- Eq. 3-7: PDF page 6;
- Eq. 8-10: PDF page 7;
- Eq. 11-16: PDF page 8;
- supplementary Eq. I1-I2 and the depth-filter profile: PDF page 13.

The preliminary transmittance equation appears to use the current element's alpha inside a product whose index is `j`; later equations use the conventional preceding-element product. SplatForge must explicitly define `T_i = product_{j<i}(1-alpha_j)` rather than preserving a notation error. Eq. 4 also omits an explicit dot-product mark in typesetting; the dimensional interpretation is the scalar inner product between the accumulated normal and the camera ray.

## Paper mathematics in SplatForge notation

The equations below are independently restated mathematical relationships. Equation references identify where the idea appears in PGSR v2; they are not copied implementation behavior.

Let a Gaussian have world-space center `m`, positive scales `s = (s0,s1,s2)`, and orientation matrix `Q`. A candidate surface normal is the orientation axis associated with the minimum scale:

```text
k   = argmin_j s[j]
n_w = sign * Q[:, k]
```

The paper penalizes the minimum scale (Eq. 1). The tie-break, orientation representation, normal sign and gradients at scale ties are not fully specified and must be owned by SplatForge.

Fix the world-to-camera convention as an eventual OpSpec decision:

```text
x_c = R_cw (x_w - C_w)
m_c = R_cw (m - C_w)
n_c = R_cw n_w
d   = dot(n_c, m_c)
```

This represents the camera-space plane `dot(n_c, x_c) = d`. For front-to-back splat weights `w_i = T_i * alpha_i`, where `T_i` is transmittance before element `i`, the paper's normal and distance accumulation (Eq. 2-3) can be written:

```text
n_bar = sum_i w_i * n_c[i]
d_bar = sum_i w_i * d[i]
```

For pixel homogeneous coordinate `p_h` and an unnormalized camera ray `q = inverse(K) @ p_h`, the plane-pair depth relationship (Eq. 4) is:

```text
z = d_bar / dot(n_bar, q)
```

The pair `(n_bar, d_bar)` is homogeneous. Do not normalize only `n_bar`; scaling both members by the same nonzero factor leaves the plane unchanged. The final spec must say whether `q.z == 1` and `z` is camera-axis depth, or whether `q` is unit length and the result is metric ray distance. It must also define grazing rays, sign, empty pixels and invalid outputs.

For a depth image, four unprojected neighboring points can define a local normal (Eq. 5):

```text
v0 = P_right - P_left
v1 = P_down  - P_up
n_depth = normalize(cross(v0, v1))
```

The exact sample positions, cross order, image-axis orientation, border behavior and zero-cross-product behavior are SplatForge decisions.

For two cameras, a plane-induced homography has the mathematical form used in Eq. 7-8:

```text
H_ref_to_near = K_near @ (R_ref_to_near - t_ref_to_near @ n_ref.T / d_ref) @ inverse(K_ref)
```

This line is not enough to implement the operator. The contract must freeze pose direction, column/row convention, plane sign, homogeneous normalization, `d` near zero, output validity and gradients.

The geometric branch applies forward and backward homographies and measures pixel-space roundtrip error (Eq. 9-10). The paper uses a hard threshold at one pixel and detaches the error-derived weight. Threshold equality, valid-count normalization, out-of-bounds handling and intermediate homogeneous normalization remain contract decisions.

The photometric branch maps a 7 by 7 grayscale patch into a neighbor frame and evaluates normalized cross-correlation (Eq. 11). Patch size 49 is a PGSR consumer profile, not the stable core API. SplatForge separates warping from ZNCC so each has one testable meaning.

## Operator classification

| Candidate semantic unit | Layer | Form | Initial provider plan | Why |
|---|---|---|---|---|
| `camera.project_pinhole@1` | core camera | operator | Torch, then optimized | Shared by points, Gaussians and meshes |
| `camera.pixel_rays@1` | core camera | operator or decomposition | Torch | Fixes pixel-center and ray normalization semantics |
| `gaussian.covariance_from_scale_rotation@1` | Gaussian | operator | Torch; fuse later only if equivalent | Gaussian-specific input decoding |
| `gaussian.project_covariance@1` | Gaussian | operator | Torch plus native CUDA candidate | Produces screen ellipses without making rasterization own camera objects |
| shortest-axis plane extraction | Gaussian | recipe first; promote only if needed | Torch | Argmin/tie/sign semantics matter, but the arithmetic alone is small |
| camera-space plane transform/offset | spatial | recipe or small operator | Torch | Not inherently Gaussian |
| `splat.rasterize_features@1` | splat | operator | high-precision reference plus native CUDA | Expensive ordered coverage/compositing; payload is representation-neutral |
| `spatial.plane_depth_from_homogeneous@1` | spatial | recipe first; promote if provider specialization is useful | Torch, possibly Triton | Small arithmetic with important invalid semantics |
| `spatial.depth_to_normal_cross@1` | spatial | operator | Torch, then Triton | Useful to depth maps from any representation |
| `camera.plane_induced_homography@1` | camera/geometry | operator | Torch | Reusable multi-view geometry |
| `sampling.patch_warp@1` | sampling | operator | Torch, then Triton | Owns interpolation, boundary and grid gradients |
| `matching.patch_zncc@1` | matching | operator | Torch, then Triton | Owns masked reduction and low-variance behavior |
| roundtrip reprojection error | matching/geometry | recipe first | Torch | Composition of homography application and norm |
| edge-aware single-view loss | recipe | Python composition | Torch primitives | Training policy, not an expensive reusable primitive |
| multi-view weights/loss mix | recipe | Python composition | Torch primitives | Hard thresholds and schedules belong to the algorithm recipe |
| exposure affine and RGB loss | recipe | Python composition | Torch primitives | Too small and policy-specific for a kernel contract |
| neighbor-frame graph | adapter/recipe | host-side logic | Python | Dataset and training policy |
| TSDF fusion and marching cubes | integration or later volume/mesh ops | external adapter first | approved dependency | Mesh production is downstream of the PGSR-style core |

## Crucial rasterization boundary

The PGSR-style geometry output does not require a permanently separate paper rasterizer. The generic operator can accumulate an explicit last-dimension payload:

```python
payload = concatenate([features, camera_normals, plane_offsets[..., None]], dim=-1)
accumulated, alpha = sf.splat.rasterize_features(
    ellipse,
    sort_depth,
    opacity,
    payload,
    image_size=(height, width),
)
```

The caller slices `accumulated` into feature, normal and distance numerators, then applies a versioned planar-depth recipe. This is the preferred first design because new linear attributes do not fork the rasterizer.

Only if profiling proves that this composition cannot meet a real workload target should SplatForge introduce an explicit fused operator such as `splat.rasterize_planar_fields@1`. That fused operator must have its own fixed output tuple, OpSpec, reference, gradient contract and conformance suite. It cannot be a dynamic `outputs={...}` mega-kernel.

## What remains outside core

The following are part of a PGSR-style training/reconstruction recipe, not runtime abstractions:

- loss weights and start iterations;
- neighbor count and pose-distance thresholds;
- image grayscale policy for a particular recipe;
- densification and pruning schedule;
- exposure-compensation enablement;
- depth filtering threshold;
- TSDF voxel size, truncation distance and mesh cleanup;
- dataset parsing and camera-convention adapters.

They may be versioned and tested, but they must not make the operator registry depend on a `PGSRScene`, `GaussianModel`, `PointCloud` or `Mesh` class.

The paper's supplementary experimental profile reports at most eight neighbors, a maximum relative angle of 30 degrees, a relative-position interval of 0.01 to 1.5, random selection of one eligible neighbor per iteration, and rejection of depth points whose ray-normal angle exceeds 80 degrees. These numbers are dataset/scene-scale recipe parameters, not portable operator defaults. They belong in a named PGSR experiment profile if reproduced at all.

## Evidence ladder

Evidence must progress in this order:

1. analytic or hand-computed fixture;
2. readable float64-capable reference;
3. property and metamorphic tests;
4. gradient comparison or finite difference away from hard thresholds;
5. optimized-provider differential tests;
6. GPU sanitizer and stream tests;
7. fixed-implementation benchmarks;
8. recipe-level synthetic integration;
9. permitted real-workload training and reconstruction;
10. mesh-quality and end-to-end acceptance.

An upstream program's output is not a normative golden oracle unless its use has been separately authorized and reviewed. Even then it should remain isolated diagnostic evidence, not input to the implementation agent.
