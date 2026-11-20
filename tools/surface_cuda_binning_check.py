"""Exact tile membership/order gates, plus GOF/AmbiSuR renderer value/VJP gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from splatforge.providers.torch.gaussian.tile_binning_cuda import tile_members_cuda
from splatforge.studio.tile_binning import tile_members


def bins_check(bounds, width=35, height=33):
    settings = dict(width=width, height=height, tile_size=16, near=0.01, far=100.0)
    expected = tile_members(bounds, **settings)
    offsets, ids = tile_members_cuda(torch.tensor(bounds, device="cuda"), **settings)
    offsets, ids = offsets.cpu().numpy(), ids.cpu().numpy()
    for i, row in enumerate(expected):
        np.testing.assert_array_equal(row, ids[offsets[i] : offsets[i + 1]])
    return {"gaussians": len(bounds), "pairs": len(ids), "exact": True}


def renderer_check(mode, candidate_backend="cuda-binned-torch-composite"):
    from splatforge.gaussian import SurfaceOperators
    from splatforge.studio.cameras import Camera
    from splatforge.studio.config import RenderConfig
    from splatforge.studio.renderers import GaussianView, renderers

    means = torch.randn(17, 3, device="cuda") * 0.1
    means[:, 2] += 2
    scene = GaussianView(
        means.requires_grad_(),
        torch.randn(17, 4, device="cuda", requires_grad=True),
        (torch.rand(17, 3, device="cuda") * 0.1 + 0.02).requires_grad_(),
        torch.full((17,), 0.2, device="cuda", requires_grad=True),
        torch.randn(17, 1, 3, device="cuda", requires_grad=True),
        0,
    )
    camera = Camera(
        torch.tensor([[30.0, 0, 16], [0, 31, 17], [0, 0, 1]], device="cuda"),
        torch.eye(4, device="cuda"),
        33,
        35,
    )
    config = RenderConfig(tile_batch_size=8)
    inputs = scene.means, scene.quats, scene.scales, scene.opacities, scene.sh
    outputs, gradients, absolute = [], [], []
    for backend in ["torch-batched-cuda-composite", candidate_backend]:
        renderer = renderers.get(backend)(config, mode=mode, ops=SurfaceOperators())
        result = renderer.render(
            scene, camera, distortions="l1" if mode == "gof" else "none"
        )
        values = [
            getattr(result, name)
            for name in [
                "rgb",
                "alpha",
                "depth",
                "normals",
                "distortion_l1",
                "distortion_l2",
                "ray_color",
            ]
        ]
        # Deterministic nonuniform adjoints, same for both paths.
        loss = sum(
            (
                x * torch.linspace(0.2, 1.0, x.numel(), device="cuda").reshape(x.shape)
            ).sum()
            for x in values
        )
        gradients.append(torch.autograd.grad(loss, inputs))
        absolute.append(result.projected_means.absgrad.clone())
        outputs.append(values)
    for a, b in zip(outputs[0], outputs[1]):
        torch.testing.assert_close(a, b, atol=3e-5, rtol=3e-5)
    for a, b in zip(gradients[0], gradients[1]):
        torch.testing.assert_close(a, b, atol=3e-4, rtol=3e-4)
    torch.testing.assert_close(*absolute, atol=3e-4, rtol=3e-4)
    return {
        "mode": mode,
        "backend": candidate_backend,
        "values_vjp_absolute_densification_gradients": "PASS",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(113)
    torch.set_num_threads(2)
    report = {"stage": "CHECKING", "bins": [], "renderers": []}

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False))

    save()
    try:
        rng = np.random.default_rng(17)
        for n in [0, 1, 33, 1000]:
            bounds = rng.normal(size=(n, 4)).astype(np.float32)
            bounds[:, :2] = bounds[:, :2] * 40 + 15
            bounds[:, 2] = np.abs(bounds[:, 2]) * 30
            bounds[:, 3] = np.round(bounds[:, 3], 1) + 2
            report["bins"].append(bins_check(bounds))
            save()
        edge = np.array(
            [
                [0, 0, 0, 2],
                [16, 16, 0, 2],
                [32, 32, 16, 2],
                [-16, 0, 16, 2],
                [51, 0, 16, 2],
                [0, 0, 16, 0.01],
                [0, 0, 16, 100],
                [0, 0, 1e20, 2],
            ],
            dtype=np.float32,
        )
        report["bins"].append(bins_check(edge))
        bad = torch.zeros(1, 4, device="cuda")
        bad[0, 0] = float("nan")
        try:
            tile_members_cuda(
                bad, width=35, height=33, tile_size=16, near=0.01, far=100
            )
        except FloatingPointError:
            report["nonfinite_rejected"] = True
        else:
            raise AssertionError("nonfinite footprint accepted")
        for mode in ["gof", "ambisur"]:
            report["renderers"].append(renderer_check(mode))
            save()
        # Wall time includes the documented scalar synchronization and allocation.
        bounds = rng.normal(size=(100000, 4)).astype(np.float32)
        bounds[:, :2] = bounds[:, :2] * 200 + 200
        bounds[:, 2] = np.abs(bounds[:, 2]) * 4
        bounds[:, 3] = np.abs(bounds[:, 3]) + 1
        settings = dict(width=777, height=581, tile_size=16, near=0.01, far=100)
        device_bounds = torch.tensor(bounds, device="cuda")
        tile_members_cuda(device_bounds, **settings)
        torch.cuda.synchronize()
        start = time.perf_counter()
        tile_members(bounds, **settings)
        report["cpu_binning_ms"] = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        tile_members_cuda(device_bounds, **settings)
        torch.cuda.synchronize()
        report["cuda_binning_ms"] = (time.perf_counter() - start) * 1000
        report["stage"] = "PASS_EXACT_BINS_AND_RENDERER_VJP"
        save()
    except Exception:
        report.update(stage="FAILED", error=traceback.format_exc())
        save()
        raise


if __name__ == "__main__":
    main()
