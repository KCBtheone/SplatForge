"""Fused event values and analytic VJP vs standalone tensor mathematics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

import torch
import torch.nn.functional as F

from splatforge.providers.torch.gaussian.events_cuda import surface_events_cuda


def reference(
    delta,
    proxy,
    q2,
    q3,
    normals,
    distance,
    opacity,
    rays,
    valid,
    *,
    mode,
    near,
    far,
    extent,
    q3_exponent=None,
):
    squared = torch.einsum("blpi,blij,blpj->blp", delta, q2, delta)
    if mode == "gof":
        qd = torch.einsum("blij,bpj->blpi", q3, rays)
        denominator = (qd * rays[:, None]).sum(-1).clamp_min(1e-12)
        depth = (proxy * qd).sum(-1) / denominator
        offset = depth[..., None] * rays[:, None] - proxy
        matrix = q3 if q3_exponent is None else q3_exponent
        exponent = torch.einsum("blpi,blij,blpj->blp", offset, matrix, offset)
        response = torch.exp(-0.5 * exponent.clamp_min(0))
        n = F.normalize(-qd, dim=-1)
    else:
        n = normals[:, :, None].expand(-1, -1, rays.shape[1], -1)
        den = (n * rays[:, None]).sum(-1)
        safe = torch.where(den.abs() > 1e-8, den, 1)
        depth = distance[:, :, None] / safe
        response = torch.where(den.abs() > 1e-8, torch.exp(-0.5 * squared), 0)
    response = torch.where(
        valid[:, :, None] & (squared <= extent**2) & (depth > near) & (depth < far),
        response,
        0,
    )
    return opacity[:, :, None] * response, depth, n


def check(mode, count, strided=False):
    B, L, P = 2, count, 7
    delta = torch.randn(B, L, P, 2, device="cuda") * 0.3
    proxy = torch.randn(B, L, P, 3, device="cuda") * 0.1
    proxy[..., 2] += 2
    q2 = torch.eye(2, device="cuda").expand(B, L, -1, -1).clone()
    q3 = torch.randn(B, L, 3, 3, device="cuda") * 0.1
    q3 = q3 @ q3.transpose(-1, -2) + torch.eye(3, device="cuda")
    normals = torch.randn(B, L, 3, device="cuda") * 0.1
    normals[..., 2] = -1
    if L > 1:
        normals[:, 0] = 0  # branch-safe parallel plane
    distance = -torch.ones(B, L, device="cuda") * 2
    opacity = torch.rand(B, L, device="cuda") * 0.5
    rays = torch.randn(B, P, 3, device="cuda") * 0.1
    rays[..., 2] = 1
    if strided:
        proxy = proxy[:, :, :1].expand(-1, -1, P, -1)
        q2 = q2.transpose(-1, -2)
    inputs = [
        x.detach().requires_grad_()
        for x in (delta, proxy, q2, q3, normals, distance, opacity, rays)
    ]
    valid = torch.rand(B, L, device="cuda") > 0.2
    settings = dict(mode=mode, near=0.01, far=100.0, extent=3.0)
    a = reference(*inputs, valid, **settings)
    b = surface_events_cuda(*inputs, valid, **settings)
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, atol=3e-5, rtol=3e-5)
    adj = [torch.randn_like(x) for x in a]
    ga = torch.autograd.grad(a, inputs, adj, allow_unused=True)
    gb = torch.autograd.grad(b, inputs, adj, allow_unused=True)
    error = 0.0
    for value, x, y in zip(inputs, ga, gb):
        x = torch.zeros_like(value) if x is None else x
        y = torch.zeros_like(value) if y is None else y
        torch.testing.assert_close(x, y, atol=3e-4, rtol=3e-4)
        if x.numel():
            error = max(error, float((x - y).abs().max()))
    return {"mode": mode, "count": L, "strided": strided, "max_vjp_error": error}


def check_truncation_boundary():
    delta = torch.tensor([[[[-95.11126708984375, 79.03306579589844]]]], device="cuda")
    delta = delta.expand(1, 1, 3, 2).clone()
    delta[0, 0, 0, 0] -= 0.00005
    delta[0, 0, 2, 0] += 0.00005
    q2 = torch.tensor(
        [
            [
                [
                    [0.00026173630612902343, 0.0000019538745164027205],
                    [0.0000019538740616553696, 0.000266027549514547],
                ]
            ]
        ],
        device="cuda",
    )
    proxy = torch.zeros(1, 1, 3, 3, device="cuda")
    q3 = torch.eye(3, device="cuda")[None, None]
    normal = torch.tensor([[[0.0, 0.0, -1.0]]], device="cuda")
    distance = torch.tensor([[-2.0]], device="cuda")
    opacity = torch.tensor([[0.1]], device="cuda")
    rays = torch.tensor([[[0.0, 0.0, 1.0]]], device="cuda").expand(1, 3, 3)
    valid = torch.ones(1, 1, device="cuda", dtype=torch.bool)
    inputs = delta, proxy, q2, q3, normal, distance, opacity, rays, valid
    settings = dict(mode="ambisur", near=0.01, far=100.0, extent=2.0)
    expected = reference(*inputs, **settings)[0]
    actual = surface_events_cuda(*inputs, **settings)[0]
    assert torch.equal(expected > 0, actual > 0)
    assert (expected > 0).any() and (expected == 0).any()
    torch.testing.assert_close(expected, actual, atol=3e-5, rtol=3e-5)
    return {
        "reference_alpha": expected.flatten().tolist(),
        "cuda_alpha": actual.flatten().tolist(),
        "support_exact": True,
    }


def check_trained_gof_quadratic():
    """Real checkpoint1367 event: flat nine-term accumulation loses ~0.009."""
    P = 256
    delta = torch.zeros(1, 1, P, 2, device="cuda")
    proxy = torch.tensor(
        [-0.2886769473552704, 0.1368797868490219, 1.8168206214904785],
        device="cuda",
    )[None, None, None].expand(1, 1, P, 3)
    q2 = torch.eye(2, device="cuda")[None, None]
    q3 = torch.tensor(
        [
            [237675696.0, -186987264.0, -131496272.0],
            [-186985680.0, 147278528.0, 103755056.0],
            [-131493384.0, 103753656.0, 73302176.0],
        ],
        device="cuda",
    )[None, None]
    normals = torch.zeros(1, 1, 3, device="cuda")
    distance = torch.zeros(1, 1, device="cuda")
    opacity = torch.tensor([[0.9999995231628418]], device="cuda")
    rays = torch.tensor(
        [-0.16388168931007385, 0.06173747777938843, 1.0], device="cuda"
    )[None, None].expand(1, P, 3)
    inputs = [
        x.detach().requires_grad_()
        for x in (delta, proxy, q2, q3, normals, distance, opacity, rays)
    ]
    valid = torch.ones(1, 1, device="cuda", dtype=torch.bool)
    settings = dict(mode="gof", near=0.01, far=100.0, extent=3.0)
    expected = reference(*inputs, valid, **settings)
    actual = surface_events_cuda(*inputs, valid, **settings)
    for left, right in zip(expected, actual):
        torch.testing.assert_close(left, right, atol=3e-5, rtol=3e-5)
    adjoints = [torch.ones_like(x) / x.numel() for x in expected]
    a = torch.autograd.grad(expected, inputs, adjoints, allow_unused=True)
    b = torch.autograd.grad(actual, inputs, adjoints, allow_unused=True)
    for value, left, right in zip(inputs, a, b):
        left = torch.zeros_like(value) if left is None else left
        right = torch.zeros_like(value) if right is None else right
        torch.testing.assert_close(left, right, atol=3e-4, rtol=3e-4)
    return {"reference_alpha": float(expected[0][0, 0, 0]), "values_vjp": "PASS"}


def check_trained_gof_background_event():
    """Captured pixel7,498 tiny alpha affects the normalized final normal."""
    P = 256
    delta = torch.zeros(1, 1, P, 2, device="cuda")
    proxy = torch.tensor(
        [0.1500987559556961, -0.44884487986564636, 2.1193692684173584],
        device="cuda",
    )[None, None, None].expand(1, 1, P, 3)
    q2 = torch.eye(2, device="cuda")[None, None]
    q3 = torch.tensor(
        [
            [55157488.0, -27473672.0, -4894686.0],
            [-27473702.0, 13882232.0, 2333097.0],
            [-4894670.0, 2333087.0, 492006.65625],
        ],
        device="cuda",
    )[None, None]
    rays = torch.tensor([0.07606324553489685, -0.196311354637146, 1.0], device="cuda")[
        None, None
    ].expand(1, P, 3)
    operands = (
        delta,
        proxy,
        q2,
        q3,
        torch.zeros(1, 1, 3, device="cuda"),
        torch.zeros(1, 1, device="cuda"),
        torch.ones(1, 1, device="cuda"),
        rays,
        torch.ones(1, 1, device="cuda", dtype=torch.bool),
    )
    settings = dict(mode="gof", near=0.01, far=100.0, extent=3.0)
    expected = reference(*operands, **settings)
    actual = surface_events_cuda(*operands, **settings)
    # Additional sensitive regression, not a relaxation of the full-image gate.
    torch.testing.assert_close(expected[0], actual[0], atol=1e-12, rtol=1e-5)
    for left, right in zip(expected[1:], actual[1:]):
        torch.testing.assert_close(left, right, atol=3e-5, rtol=3e-5)
    return {"reference_alpha": float(expected[0][0, 0, 0]), "values": "PASS"}


def check_indexed_gof_q3():
    """Independent Qray/exponent gathers retain separate parent adjoints."""
    base = (torch.eye(3, device="cuda")[None].repeat(3, 1, 1) * 0.02).requires_grad_()
    precision = torch.linalg.inv(base @ base.transpose(-1, -2))
    ids = torch.tensor([[0, 1, 0, 2, 1]], device="cuda")
    B, L, P = 1, 5, 7
    rays = torch.randn(B, P, 3, device="cuda") * 0.001
    rays[..., 2] = 1
    proxy = torch.zeros(B, L, P, 3, device="cuda")
    proxy[..., 2] = 2
    delta = torch.zeros(B, L, P, 2, device="cuda")
    q2 = torch.eye(2, device="cuda").expand(B, L, 2, 2)
    normals = torch.zeros(B, L, 3, device="cuda")
    distance = torch.zeros(B, L, device="cuda")
    opacity = torch.full((B, L), 0.7, device="cuda")
    valid = torch.ones(B, L, device="cuda", dtype=torch.bool)
    settings = dict(mode="gof", near=0.01, far=100.0, extent=3.0)

    def evaluate(function):
        return function(
            delta,
            proxy,
            q2,
            precision[ids],
            normals,
            distance,
            opacity,
            rays,
            valid,
            q3_exponent=precision[ids],
            **settings,
        )

    expected = evaluate(reference)
    actual = evaluate(surface_events_cuda)
    for left, right in zip(expected, actual):
        torch.testing.assert_close(left, right, atol=3e-5, rtol=3e-5)
    adjoints = [torch.randn_like(value) for value in expected]
    left = torch.autograd.grad(expected, base, adjoints, retain_graph=True)[0]
    right = torch.autograd.grad(actual, base, adjoints)[0]
    torch.testing.assert_close(left, right, atol=3e-4, rtol=3e-3)
    return {"values_parent_inverse_vjp": "PASS", "duplicate_gather_indices": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(31)
    report = {"stage": "CHECKING", "events": [], "renderers": []}

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False))

    save()
    try:
        for mode in ["gof", "ambisur"]:
            for count in [0, 1, 5, 33]:
                for strided in [False, True]:
                    report["events"].append(check(mode, count, strided))
                    save()
        from surface_cuda_binning_check import renderer_check

        report["truncation_boundary"] = check_truncation_boundary()
        report["trained_gof_quadratic"] = check_trained_gof_quadratic()
        report["trained_gof_background_event"] = check_trained_gof_background_event()
        report["indexed_q3_inverse"] = check_indexed_gof_q3()
        save()

        for mode in ["gof", "ambisur"]:
            report["renderers"].append(
                renderer_check(mode, "cuda-binned-events-composite")
            )
            save()
        report["stage"] = "PASS_EVENTS_AND_RENDERER_VALUE_VJP"
        save()
    except Exception:
        report.update(stage="FAILED", error=traceback.format_exc())
        save()
        raise


if __name__ == "__main__":
    main()
