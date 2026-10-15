# matching.patch_zncc@1 pre-freeze review questions

Status: blocking review input, not an OpSpec.

The proposal is mathematically clear for ordinary finite patches, but several
clauses are not yet jointly implementable or precise enough for a frozen GPU
provider task.

## 1. Non-finite values versus asynchronous CUDA

The proposal currently says non-finite inputs are rejected. A CUDA provider
cannot inspect tensor values and raise a synchronous structured Python error
without a device-to-host decision or another synchronization mechanism. That
conflicts with metadata-only dispatch and the no-host-synchronization cell.

The spec owner must choose one portable semantic rule:

- non-finite values are outside the contract domain, with no runtime rejection
  guarantee;
- any unmasked non-finite sample makes that patch invalid with zero score and
  zero input gradients;
- the operator has an explicitly synchronizing validation mode or a different
  schema.

Masked-value irrelevance must state whether it includes masked NaN and Inf.

## 2. Layout grammar for a flattened patch launch

The phrase last-dimension contiguous only guarantees a final stride of one.
Prefix axes can still be transposed, sliced, overlapping, or otherwise
non-dense. A kernel cannot map a flat patch index to a base address using
patch_index times sample_count for all such tensors.

The first Triton cell must freeze one of:

- fully C-contiguous lhs, rhs, and mask;
- a formally defined dense-prefix layout that can be viewed as [P, 49] without
  a copy;
- bounded-rank prefix strides passed to and handled by the kernel.

The fixed provider must reject other layouts before launch.

## 3. Finite range and overflow

For arbitrary finite fp32 values, the mean sum or squared centered sums can
overflow even though the real-number ZNCC is finite. Therefore the current
property that finite valid inputs produce finite outputs needs either:

- a documented input magnitude domain;
- a scaled or Welford-style stable semantic evaluation requirement; or
- case-class-specific behavior and tests for arithmetic overflow.

The chosen rule affects the reference, analytic fixtures, Triton reduction,
gradient equations, and tolerances.

## 4. Variance epsilon representation

The proposed public schema takes variance_epsilon explicitly, so a default is
not required unless a convenience wrapper adds one. The contract still must
freeze:

- Python scalar versus zero-dimensional tensor;
- conversion to fp32 or fp64 accumulator dtype;
- representability and overflow behavior during conversion;
- strict greater-than comparison and equality behavior;
- whether the value is part of the compile/static specialization key.

## 5. Backward input layouts and second order

The first provider must state whether arbitrary legal grad_score strides,
including expanded stride-zero tensors, are consumed directly. If a copy is
allowed, it must be planned and measured rather than introduced inside
backward.

The repository also needs one named structured error and a tested point at
which a second-order request raises. Merely returning detached first-order
gradients is not an explicit unsupported result.

## 6. Compile, fake, export, and autocast

The first work order must list each capability as supported or unsupported.
The phrase run declared tests is insufficient unless the manifest freezes the
declaration. Autocast-active calls likewise need an exact cast or rejection
rule.

These decisions must be resolved in the independent spec/reference change,
then reflected consistently in the OpSpec, Torch reference, fixtures, public
tests, implementation manifest template, tolerance table, and work order.

