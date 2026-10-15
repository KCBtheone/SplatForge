# `testing.scale_shift@1`

This is a deliberately non-domain Phase 0 fixture, not a public scientific
operator. It exists to prove that a specification revision can be registered,
that a separately manifested implementation can be selected using tensor
metadata, and that the exact implementation can produce a structured report.

For a floating PyTorch tensor `x` and finite real static attributes `scale` and
`bias`, the result is `x * scale + bias`. The result has the same shape, dtype,
and device as `x`; it is freshly produced and does not alias or mutate `x`.
Empty tensors are valid. Tensor non-finite values follow ordinary PyTorch
arithmetic. Non-finite static attributes are rejected before dispatch.

The Phase 0 Torch implementation declares eager CPU support for float32 and
float64 contiguous tensors. It is intentionally not release-qualified.

