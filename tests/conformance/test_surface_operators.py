from __future__ import annotations

import unittest
from dataclasses import replace

import torch

from splatforge.core.errors import SchemaValidationError
from splatforge.gaussian import SurfaceOperators, create_surface_runtime


class SurfaceOperatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.ops = SurfaceOperators()

    def test_covariance_rotates_anisotropic_axes(self):
        q = torch.tensor([[2**-0.5, 0.0, 0.0, 2**-0.5]], dtype=torch.float64)
        s = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
        result = self.ops("gaussian.covariance@1", q, s)
        torch.testing.assert_close(
            result, torch.diag(torch.tensor([4.0, 1.0, 9.0], dtype=torch.float64))[None]
        )

    def test_explicit_batch_composite_and_ray_color_match_unbatched_contracts(self):
        torch.manual_seed(12)
        alpha = torch.rand(3, 7, 5, dtype=torch.float64, requires_grad=True) * 0.2
        color = torch.rand(3, 7, 3, dtype=torch.float64, requires_grad=True)
        depth = torch.rand(3, 7, 5, dtype=torch.float64, requires_grad=True) + 1
        normal = torch.rand(3, 7, 5, 3, dtype=torch.float64, requires_grad=True)
        batch = self.ops("splat.alpha_composite@2", alpha, color, depth, normal)
        for i in range(3):
            expected = self.ops(
                "splat.alpha_composite@1", alpha[i], color[i], depth[i], normal[i]
            )
            for actual, reference in zip(batch, expected):
                torch.testing.assert_close(actual[i], reference)
        variance = self.ops("gaussian.ray_color_variance@2", batch[-1], color, batch[0])
        for i in range(3):
            torch.testing.assert_close(
                variance[i],
                self.ops(
                    "gaussian.ray_color_variance@1", batch[-1][i], color[i], batch[0][i]
                ),
            )
        grads = torch.autograd.grad(
            variance.sum(), (alpha, color, depth, normal), allow_unused=True
        )
        self.assertIsNone(grads[0])
        self.assertIsNone(grads[2])
        self.assertIsNone(grads[3])
        self.assertTrue(torch.isfinite(grads[1]).all())

    def test_explicit_empty_batch_composite(self):
        result = self.ops(
            "splat.alpha_composite@2",
            torch.empty(2, 0, 3),
            torch.empty(2, 0, 3),
            torch.empty(2, 0, 3),
            torch.empty(2, 0, 3, 3),
        )
        for value in result:
            self.assertTrue(torch.isfinite(value).all())
            self.assertEqual(float(value.sum()), 0)

    def test_peak_and_normal_analytic(self):
        m = torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float64)
        q = torch.eye(3, dtype=torch.float64)[None]
        origin = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        ray = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
        t, response, normal = self.ops("gaussian.ray_peak@1", m, q, origin, ray)
        self.assertEqual(float(t[0, 0]), 2.0)
        self.assertEqual(float(response[0, 0]), 1.0)
        torch.testing.assert_close(
            normal[0, 0], torch.tensor([0.0, 0.0, -1.0], dtype=torch.float64)
        )

    def test_ray_peak_gradient_check(self):
        m = torch.tensor([[0.1, -0.2, 2.0]], dtype=torch.float64, requires_grad=True)
        q = torch.diag(torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64))[
            None
        ].requires_grad_()
        origin = torch.zeros((2, 3), dtype=torch.float64)
        rays = torch.tensor([[0.1, 0.2, 1.0], [-0.1, 0.2, 1.0]], dtype=torch.float64)
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda a, b: self.ops("gaussian.ray_peak@1", a, b, origin, rays),
                (m, q),
                atol=1e-5,
                rtol=1e-3,
            )
        )

    def test_pixel_centers_and_plane_normal(self):
        K = torch.tensor([[4.0, 0, 2.0], [0, 4.0, 2.0], [0, 0, 1.0]])
        rays = self.ops("camera.pixel_rays@1", K, width=4, height=4)
        torch.testing.assert_close(rays[0, 0], torch.tensor([-0.375, -0.375, 1.0]))
        normal, valid = self.ops("geometry.depth_normals@1", torch.full((4, 4), 2.0), K)
        self.assertEqual(int(valid.sum()), 4)
        torch.testing.assert_close(
            normal[valid], torch.tensor([[0.0, 0.0, -1.0]]).expand(4, -1)
        )

    def test_distortions_match_pairwise_definition(self):
        alpha = torch.tensor(
            [[0.2, 0.3], [0.4, 0.5], [0.6, 0.1]],
            dtype=torch.float64,
            requires_grad=True,
        )
        depth = torch.tensor(
            [[3.0, 2.0], [1.0, 4.0], [2.0, 1.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        out = self.ops(
            "splat.alpha_composite@1",
            alpha,
            torch.ones(3, 3, dtype=torch.float64),
            depth,
            torch.zeros(3, 2, 3, dtype=torch.float64),
        )
        weights = out[-1]
        difference = depth[:, None] - depth[None]
        l1 = (weights[:, None] * weights[None] * difference.abs()).sum((0, 1))
        l2 = (
            weights.detach()[:, None] * weights.detach()[None] * difference.square()
        ).sum((0, 1))
        torch.testing.assert_close(out[4], l1)
        torch.testing.assert_close(out[5], l2)
        out[5].sum().backward()
        self.assertIsNone(alpha.grad)
        self.assertIsNotNone(depth.grad)

    def test_ray_color_only_updates_colors(self):
        w = torch.tensor([[0.2, 0.3], [0.5, 0.4]], requires_grad=True)
        color = torch.tensor([[0.2, 0.4, 0.1], [0.6, 0.3, 0.9]], requires_grad=True)
        composite = torch.zeros(2, 3, requires_grad=True)
        self.ops("gaussian.ray_color_variance@1", w, color, composite).sum().backward()
        self.assertIsNone(w.grad)
        self.assertIsNone(composite.grad)
        self.assertGreater(float(color.grad.abs().sum()), 0)

    def test_opacity_field_plateau_and_camera_minimum(self):
        points = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]])
        means = torch.tensor([[0.0, 0.0, 2.0]])
        inverse = torch.eye(3)[None]
        alpha = torch.tensor([0.8])
        origins = torch.tensor([[0.0, 0.0, 0.0]])
        values = self.ops(
            "gaussian.opacity_field@1",
            points,
            means,
            inverse,
            alpha,
            origins,
            gaussian_chunk=1,
        )
        torch.testing.assert_close(
            values, torch.tensor([0.8 * torch.exp(torch.tensor(-0.5)), 0.8, 0.8])
        )
        two_views = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 4.0]])
        mirrored = self.ops(
            "gaussian.opacity_field@1", points, means, inverse, alpha, two_views
        )
        torch.testing.assert_close(mirrored, torch.tensor([values[0], 0.8, values[0]]))

    def test_empty_gaussians_and_low_variance_patch(self):
        output = self.ops(
            "splat.alpha_composite@1",
            torch.empty(0, 2),
            torch.empty(0, 3),
            torch.empty(0, 2),
            torch.empty(0, 2, 3),
        )
        self.assertEqual(float(output[1].sum()), 0.0)
        correlation, valid = self.ops(
            "matching.patch_zncc@1",
            torch.ones(2, 7),
            torch.ones(2, 7),
            torch.ones(2, 7),
        )
        self.assertFalse(valid.any())
        self.assertEqual(float(correlation.sum()), 0.0)

    def test_shape_dtype_and_unknown_attribute_rejection(self):
        with self.assertRaises(SchemaValidationError):
            self.ops("gaussian.covariance@1", torch.ones(1, 3), torch.ones(1, 3))
        with self.assertRaises(SchemaValidationError):
            self.ops(
                "gaussian.covariance@1",
                torch.ones(1, 4),
                torch.ones(1, 3, dtype=torch.float64),
            )
        with self.assertRaises(SchemaValidationError):
            self.ops(
                "camera.pixel_rays@1", torch.eye(3), width=2, height=2, unexpected=True
            )

    def test_sh_dc_and_ambiguity_independence(self):
        coefficients = torch.zeros(2, 16, 3)
        coefficients[:, 0] = 1
        score = self.ops("gaussian.sh_ambiguity@1", coefficients)
        torch.testing.assert_close(score, torch.zeros(2))
        color = self.ops(
            "gaussian.sh_color@1",
            coefficients,
            torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
            degree=3,
        )
        torch.testing.assert_close(color, torch.full((2, 3), 0.7820947918))

    def test_explicit_provider_extension_never_falls_back(self):
        runtime = create_surface_runtime(seal=False)
        original = runtime.implementations.get(
            "splatforge.torch.gaussian.covariance.v1"
        )
        manifest = replace(
            original.manifest,
            implementation_id="test.explicit.covariance",
            provider="test-provider",
        )

        def failing(*_):
            raise RuntimeError("provider-sentinel")

        runtime.register_implementation(manifest, failing)
        runtime.seal()
        ops = SurfaceOperators(
            runtime,
            implementations={"gaussian.covariance@1": manifest.implementation_id},
        )
        with self.assertRaisesRegex(RuntimeError, "provider-sentinel"):
            ops(
                "gaussian.covariance@1",
                torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
                torch.ones(1, 3),
            )


if __name__ == "__main__":
    unittest.main()
