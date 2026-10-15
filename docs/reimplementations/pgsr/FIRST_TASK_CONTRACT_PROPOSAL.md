# First task contract proposal: `matching.patch_zncc@1`

Status: proposal for an independent spec/reference review. It is not frozen and must not be used to dispatch the draft work order.

## Why this is the first task

PGSR's multi-view photometric branch evaluates NCC over a 7 by 7 grayscale patch. A masked ZNCC reduction is directly relevant, but it does not require the implementation agent to understand Gaussian representations, camera poses or rasterization. It is large enough to test GPU reductions and autograd while remaining small enough for analytic and metamorphic verification.

Warp and ZNCC are deliberately separate:

```text
image + sampling grid --sampling.patch_warp--> sampled patches + mask
sampled patches + mask --matching.patch_zncc--> score + valid
```

This resolves an important API boundary: grayscale conversion, interpolation, out-of-bounds handling and grid gradients belong to `patch_warp` or the calling recipe, not to `patch_zncc`.

## Proposed public schema

```python
score, valid = sf.matching.patch_zncc(
    lhs,                  # floating Tensor [..., S]
    rhs,                  # floating Tensor [..., S]
    sample_mask,          # bool Tensor [..., S]
    variance_epsilon,     # finite non-negative semantic scalar
)
```

Proposed observable rules:

- `lhs`, `rhs` and `sample_mask` have exactly the same shape; no broadcasting.
- `lhs` and `rhs` have the same floating dtype and device.
- The final dimension contains samples. Every prefix element is an independent patch.
- The public Torch implementation may support arbitrary `S`, including zero. `S=49` is only the first Triton support cell.
- Inputs are finite. Non-finite inputs are rejected rather than silently sanitized.
- The operation does not mutate or alias inputs.
- Mask is nondifferentiable.

For one patch, with `m_i` in `{0,1}`:

```text
n   = sum_i m_i
mx  = sum_i (m_i * x_i) / n
my  = sum_i (m_i * y_i) / n
dx_i = x_i - mx
dy_i = y_i - my
sxx = sum_i (m_i * dx_i^2)
syy = sum_i (m_i * dy_i^2)
sxy = sum_i (m_i * dx_i * dy_i)

valid = (n >= 2) and (sxx > variance_epsilon) and (syy > variance_epsilon)
score = sxy / sqrt(sxx * syy)      when valid
score = 0                           otherwise
```

`variance_epsilon` is compared with the centered sum, not a population/sample-normalized variance. Its exact default and dtype conversion must be calibrated and frozen before dispatch. Equality is proposed to be invalid.

Do not clamp the valid score to `[-1,1]`; the mathematical range is enforced by tests with a numeric tolerance. A clamp would introduce an extra gradient boundary.

## Proposed dtype and gradient contract

- fp16 and fp32 provider cells accumulate `n`, means and centered sums in fp32 and return fp32 score.
- A CPU fp64 reference path preserves fp64 for high-precision tests.
- `valid` is bool.
- Gradients with respect to valid `lhs` and `rhs` follow the stated expression.
- Invalid patches have exactly zero gradient for both inputs.
- Mask, valid branch decisions and threshold comparisons are nondifferentiable.
- The first Triton provider must support first-order VJP for both inputs.
- Second-order differentiation is outside its declared cell and must raise the repository's structured unsupported error, not return a silently wrong value.
- Returned input gradients use the corresponding input dtype.

The final OpSpec must decide whether an expanded/non-contiguous upstream gradient is supported directly, copied under an explicit copy policy, or rejected. It cannot be silently materialized.

## First Triton implementation cell

```yaml
device: cuda
sample_count: 49
input_dtype: [float16, float32]
accumulation_dtype: float32
score_dtype: float32
input_layout: last_dimension_contiguous
autograd: first_order_vjp
deterministic: exact implementation plus exact environment
```

All other device, dtype, sample-count and layout cells are unsupported by this implementation ID. A caller that explicitly selects it receives a structured rejection. Auto-policy may choose another provider, but that is tested at the runtime layer and is not a kernel fallback.

## Analytic fixtures

At minimum, freeze fixtures for:

1. `x == y` with nonzero variance: score `+1`.
2. `y = a*x+b`, `a>0`: score `+1` up to tolerance.
3. `y = a*x+b`, `a<0`: score `-1` up to tolerance.
4. Orthogonal zero-mean vectors: score `0`.
5. Full mask and several partial masks.
6. Exactly two valid, nonconstant samples.
7. One valid sample and no valid samples: invalid, zero score and zero gradients.
8. Constant lhs, constant rhs and both constant: invalid.
9. Values immediately on each side of the frozen variance threshold.
10. Replacing masked elements with very large finite values: no output or gradient change.
11. One batch containing valid and invalid patches.
12. Empty prefix batch and a single patch.

Avoid using outputs from a PGSR executable as golden tensors.

The earlier SplatAtlasVol raw `plane_depth` preparation/record is also excluded:
it was not revalidated for this effort, may be stale, and is neither a semantic
source nor a fixture, golden, tolerance-calibration input, correctness oracle,
or performance baseline.

## Property and adversarial matrix

Properties away from validity thresholds:

- symmetry: `zncc(x,y) == zncc(y,x)`;
- positive affine invariance;
- negative scaling flips only the sign;
- batched evaluation equals itemwise evaluation;
- masked values and masked gradients are irrelevant;
- finite valid inputs produce finite outputs;
- valid score remains within a frozen floating-point tolerance of `[-1,1]`;
- repeated calls are bitwise equal if the exact implementation cell declares deterministic behavior.

Shape/performance coverage should include patch counts near launch tails and large workloads, for example `0, 1, 31, 32, 33, 255, 256, 257, 65536, 262144`, with mask densities near `0%, 2%, 10%, 70%, 100%`.

Gradient verification:

- fp64 Torch reference `gradcheck` on valid samples far from thresholds;
- Triton fp32 forward and VJP differential comparison with the reference;
- independent directional finite differences;
- separate lhs/rhs gradients;
- exactly zero invalid and masked gradients;
- declared upstream-gradient stride cases;
- explicit second-order rejection.

## Candidate tolerances for calibration

These are starting hypotheses, not permission for an implementation PR to choose or change thresholds:

| Cell | Forward atol/rtol | VJP atol/rtol |
|---|---:|---:|
| fp32 | `2e-5` | `1e-4` |
| fp16 with fp32 accumulation | `2e-3` | `5e-3` |

A protected calibration change must run the reference, analytic fixtures and target hardware before freezing these values.

## Candidate RTX 4080 SUPER benchmark profile

Freeze a trusted baseline before dispatch. A useful first profile is:

- patch counts 65,536 and 262,144;
- `S=49`;
- fp16 and fp32;
- mask densities 70% and 100%;
- forward and forward-plus-backward measured separately;
- fastest correct Torch eager/compiled candidate recorded;
- compile/cold cost separated from warm steady state;
- CUDA-event timing with explicit synchronization outside the measured region;
- 50 warmups and at least 200 retained samples per case;
- raw samples plus GPU, SM, driver, CUDA, Torch, Triton, clocks/temperature and build digest.

Initial performance hypotheses to calibrate, not yet gates:

- geometric-mean forward-plus-backward speedup of at least 1.25x on the large primary cases;
- no primary case slower than the fastest correct Torch baseline by more than 5% after the frozen noise rule;
- peak allocated memory no more than 1.10x the fastest correct Torch baseline;
- no `.cpu()`, `.item()`, host synchronization or undeclared contiguous copy in the implementation path.

Small cases should initially be observed for catastrophic regression rather than forced to beat launch-overhead-dominated Torch calls.

## GPU safety checks

- Compute Sanitizer memcheck on empty, tail, mixed-validity and large-index cases.
- Race/init/sync checks where supported by the kernel structure.
- No invalid launch for an empty batch.
- Explicit 32/64-bit indexing limit in the implementation manifest.
- Current PyTorch stream, no private stream and no global synchronize.
- No persistent global scratch tensor.
- Finite, declared specialization set; inputs cannot trigger unbounded autotune/JIT variants.

## Freeze questions still open

1. Exact `variance_epsilon` value and whether it is represented in input or accumulation dtype.
2. Complete Torch dtype matrix, including whether bf16 belongs in semantic v1 or a later implementation cell.
3. Upstream-gradient stride/copy policy.
4. Exact structured error for second-order requests.
5. Final tolerance calibration.
6. Trusted baseline and noise rule on the actual RTX 4080 SUPER environment.
7. Whether `torch.compile`, FakeTensor and export are required in the first candidate cell or are separately declared capabilities.

Until these questions, all digests and the base commit are frozen, `SF-PGSR-001-ZNCC-TRITON.yaml` must remain `draft_not_executable`.
