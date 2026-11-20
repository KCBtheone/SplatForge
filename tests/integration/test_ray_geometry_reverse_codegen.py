from pathlib import Path

import pytest
import torch

from tools.generate_ray_geometry_reverse import Graph, equations, generate
from splatforge.providers.torch.gaussian.radegs import regularized_ray_geometry


def test_generated_header_reproducible():
    path = (
        Path(__file__).resolve().parents[2]
        / "src/splatforge/providers/torch/gaussian/ray_geometry_reverse.cuh"
    )
    assert path.read_text() == generate()[0]


@pytest.mark.parametrize("extreme", [False, True])
def test_scalar_graph_matches_independent_cholesky_reference(extreme):
    torch.manual_seed(93)
    means = torch.randn(4, 3, dtype=torch.float64) * 0.3
    means[:, 2] += 2
    quats = torch.randn(4, 4, dtype=torch.float64)
    scales = torch.rand(4, 3, dtype=torch.float64) * 0.1 + 0.003
    if extreme:
        scales[0] = scales.new_tensor([1e-9, 50.0, 0.002])
        means[1, 0] = 100  # FOV clamp and its derivative.
    inputs = [x.requires_grad_() for x in [means, quats, scales]]
    T = torch.eye(4, dtype=torch.float64)
    T[:3, 3] = T.new_tensor([0.1, -0.2, 0.3])
    K = T.new_tensor([[500.0, 0, 388.0], [0, 501.0, 290.0], [0, 0, 1.0]])
    g = Graph()
    _, outputs = equations(g)
    values = []
    env = dict(
        means=means.T,
        quats=quats.T,
        scales=scales.T,
        T=T.flatten(),
        K=K.flatten(),
        W=777,
        H=581,
        relative=1e-6,
        absolute=1e-8,
    )
    for op, args, _ in g.nodes:
        if op == "constant":
            value = means.new_tensor(args[0])
        elif op in ("variable", "fixed"):
            value = torch.as_tensor(
                eval(args[0], {"__builtins__": {}}, env), dtype=torch.float64
            )
        elif op == "add":
            value = values[args[0]] + values[args[1]]
        elif op == "sub":
            value = values[args[0]] - values[args[1]]
        elif op == "mul":
            value = values[args[0]] * values[args[1]]
        elif op == "div":
            value = values[args[0]] / values[args[1]]
        elif op == "sqrt":
            value = values[args[0]].sqrt()
        elif op == "neg":
            value = -values[args[0]]
        else:
            a, b = [values[i] for i in args]
            value = torch.where(a >= b if op == "max" else a <= b, a, b)
        values.append(value)
    actual = torch.stack([values[x.i] for x in outputs], -1)
    d, p, n = regularized_ray_geometry(*inputs, T, K, width=777, height=581)
    expected = torch.cat([d[:, None], p, n], -1)
    torch.testing.assert_close(actual, expected, atol=2e-8, rtol=2e-8)
    adj = torch.randn_like(expected)
    left = torch.autograd.grad(actual, inputs, adj, retain_graph=True)
    right = torch.autograd.grad(expected, inputs, adj)
    for a, b in zip(left, right):
        torch.testing.assert_close(a, b, atol=2e-7, rtol=2e-7)
