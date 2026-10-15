from __future__ import annotations

import unittest
from unittest.mock import Mock

import torch

from splatforge.gaussian import SurfaceOperators
from splatforge.integrations.gsplat_radegs_moments import GsplatRadeMomentRenderer
from splatforge.studio.cameras import Camera
from splatforge.studio.config import RenderConfig
from splatforge.studio.renderers import GaussianView


class RegularizedRayGeometryTests(unittest.TestCase):
    def setUp(self):
        self.ops = SurfaceOperators()

    def call(self, means, quats, scales):
        transform = torch.eye(4, dtype=means.dtype)
        K = torch.tensor([[20, 0, 8], [0, 20, 8], [0, 0, 1]], dtype=means.dtype)
        return self.ops(
            "gaussian.regularized_ray_geometry@1",
            means,
            quats,
            scales,
            transform,
            K,
            width=16,
            height=16,
        )

    def test_isotropic_geometry_has_zero_slopes_and_faces_camera(self):
        means = torch.tensor([[0.2, -0.1, 2.0]], dtype=torch.float64)
        quats = torch.tensor([[1.0, 0, 0, 0]], dtype=torch.float64)
        distance, plane, normal = self.call(means, quats, torch.ones_like(means) * 0.1)
        torch.testing.assert_close(distance, torch.linalg.vector_norm(means, dim=-1))
        torch.testing.assert_close(plane, torch.zeros_like(plane), atol=1e-14, rtol=0)
        torch.testing.assert_close(normal, -means / distance[:, None])

    def test_cholesky_geometry_gradcheck(self):
        means = torch.tensor([[0.2, 0.1, 2.0]], dtype=torch.float64, requires_grad=True)
        quats = torch.tensor(
            [[0.8, 0.2, -0.1, 0.3]], dtype=torch.float64, requires_grad=True
        )
        scales = torch.tensor(
            [[0.01, 0.06, 0.03]], dtype=torch.float64, requires_grad=True
        )
        self.assertTrue(torch.autograd.gradcheck(self.call, (means, quats, scales)))

    def test_extreme_anisotropy_keeps_float32_outputs_and_gradients_finite(self):
        means = torch.tensor([[0.2, 0.1, 2.0], [0.3, -0.2, 3.0]], requires_grad=True)
        quats = torch.tensor(
            [[0.7, 0.3, -0.1, 0.2], [0.6, 0.5, 0.1, -0.3]], requires_grad=True
        )
        scales = torch.tensor(
            [[1e-9, 50.0, 2e-3], [1e-5, 0.06, 0.002]], requires_grad=True
        )
        outputs = self.call(means, quats, scales)
        sum(x.square().sum() for x in outputs).backward()
        for value in (*outputs, means.grad, quats.grad, scales.grad):
            self.assertEqual(value.dtype, torch.float32)
            self.assertTrue(torch.isfinite(value).all())

    def test_stable_adapter_accepts_five_output_non_rade_projection_api(self):
        renderer = GsplatRadeMomentRenderer.__new__(GsplatRadeMomentRenderer)
        renderer.stable_geometry = True
        renderer.config = RenderConfig()
        renderer.ops = self.ops
        renderer.project = Mock(
            return_value=(
                torch.ones(1, 1, 2, dtype=torch.int32),
                torch.zeros(1, 1, 2),
                torch.ones(1, 1),
                torch.ones(1, 1, 3),
                None,
            )
        )
        scene = GaussianView(
            torch.tensor([[0.0, 0.0, 2.0]]),
            torch.tensor([[1.0, 0, 0, 0]]),
            torch.tensor([[0.1, 0.1, 0.1]]),
            torch.tensor([0.1]),
            torch.zeros(1, 1, 3),
            0,
        )
        camera = Camera(
            torch.tensor([[20.0, 0, 8], [0, 20.0, 8], [0, 0, 1.0]]),
            torch.eye(4),
            16,
            16,
        )
        outputs = renderer.geometry(scene, camera)
        self.assertEqual(len(outputs), 8)
        self.assertFalse(renderer.project.call_args.kwargs["radegs"])
        torch.testing.assert_close(outputs[-3], torch.tensor([[2.0]]))
        torch.testing.assert_close(outputs[-2], torch.zeros(1, 1, 2))
        torch.testing.assert_close(outputs[-1], torch.tensor([[[0.0, 0.0, -1.0]]]))
