"""Independent value/VJP gates and paired CUDA-event timings; no training writes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

import torch

from splatforge.providers.torch.gaussian.radegs import (
    regularized_ray_geometry as reference,
)
from splatforge.providers.torch.gaussian.ray_geometry_cuda import (
    regularized_ray_geometry_cuda as candidate,
)


def fixture(n, extreme=False, cameras=False):
    means = torch.randn(n, 3, device="cuda") * 0.3
    means[:, 2] += 2
    quats = torch.randn(n, 4, device="cuda")
    scales = torch.rand(n, 3, device="cuda") * 0.1 + 0.003
    if extreme and n:
        scales[0] = scales.new_tensor([1e-9, 50.0, 0.002])
    transform = torch.eye(4, device="cuda")
    transform[0, 3] = 0.1
    # Nonidentity camera rotation tests the full covariance chain.
    angle = 0.21
    import math

    transform[:3, :3] = transform.new_tensor(
        [
            [math.cos(angle), 0, math.sin(angle)],
            [0, 1, 0],
            [-math.sin(angle), 0, math.cos(angle)],
        ]
    )
    K = means.new_tensor([[600, 0, 320], [0, 610, 240], [0, 0, 1]])
    inputs = [means, quats, scales, transform, K]
    for i, tensor in enumerate(inputs):
        tensor.requires_grad_(i < 3 or cameras)
    return inputs


def check(n, extreme=False, cameras=False):
    inputs = fixture(n, extreme, cameras)
    a = reference(*inputs, width=640, height=480)
    b = candidate(*inputs, width=640, height=480)
    errors = []
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right, atol=3e-5, rtol=3e-5)
        errors.append(float((left - right).abs().max()) if left.numel() else 0)
    adjoints = [torch.randn_like(x) for x in a]
    parameters = inputs if cameras else inputs[:3]
    ga = torch.autograd.grad(a, parameters, adjoints, retain_graph=True)
    gb = torch.autograd.grad(b, parameters, adjoints)
    gradient_errors = []
    for left, right in zip(ga, gb):
        torch.testing.assert_close(left, right, atol=3e-4, rtol=3e-4)
        gradient_errors.append(float((left - right).abs().max()) if left.numel() else 0)
    return {
        "n": n,
        "extreme": extreme,
        "camera_gradients": cameras,
        "max_value_error": max(errors),
        "max_vjp_error": max(gradient_errors),
    }


def timing(function, inputs, repeats=10):
    # Both implementations measured on exactly the same resident inputs.
    def run():
        output = function(*inputs, width=777, height=581)
        torch.autograd.grad(sum(x.sum() for x in output), inputs[:3])

    for _ in range(2):
        run()
    torch.cuda.synchronize()
    start, stop = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    for _ in range(repeats):
        run()
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) / repeats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=16384)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(1729)
    report = {
        "stage": "BUILD_AND_CHECK",
        "rows": [],
        "scope": "operator fixtures, not full method or CD acceptance",
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False))

    save()
    try:
        for n in (0, 1, 33, 257):
            for cameras in (False, True):
                report["rows"].append(check(n, cameras=cameras))
                save()
        report["rows"].append(check(33, extreme=True))
        # Discrete mask must suppress invalid z without evaluating its arithmetic.
        inputs = fixture(3)
        with torch.no_grad():
            inputs[0][1, 2] = 0
        mask = torch.tensor([True, False, True], device="cuda")
        outputs = candidate(*inputs, visible=mask, width=640, height=480)
        gradients = torch.autograd.grad(sum(x.sum() for x in outputs), inputs[:3])
        assert all(torch.isfinite(x).all() for x in (*outputs, *gradients))
        assert all((x[1] == 0).all() for x in (*outputs, *gradients))
        inputs = fixture(args.count)
        report["reference_forward_backward_ms"] = timing(reference, inputs)
        report["cuda_forward_backward_ms"] = timing(candidate, inputs)
        report["speedup"] = (
            report["reference_forward_backward_ms"] / report["cuda_forward_backward_ms"]
        )
        report["timing_count"] = args.count
        report["stage"] = "PASS_VALUE_VJP_AND_MASK"
        save()
    except Exception:
        report.update(stage="FAILED", error=traceback.format_exc())
        save()
        raise


if __name__ == "__main__":
    main()
