from __future__ import annotations

import math
import unittest

import torch

from unbiased_gsplat.ray_splat_depth import ray_splat_depth_and_weight


def transform_for_uv(
    pixel_x: float,
    pixel_y: float,
    local_u: float,
    local_v: float,
    w_row: tuple[float, float, float],
) -> torch.Tensor:
    w = torch.tensor(w_row, dtype=torch.float64)
    h_u = torch.tensor([-1.0, 0.0, local_u], dtype=torch.float64)
    h_v = torch.tensor([0.0, -1.0, local_v], dtype=torch.float64)
    u = pixel_x * w - h_u
    v = pixel_y * w - h_v
    return torch.stack((u, v, w))


class RaySplatDepthTests(unittest.TestCase):
    def test_center_hit_has_unit_weight_and_center_depth(self) -> None:
        transform = transform_for_uv(0.5, 0.5, 0.0, 0.0, (0.0, 0.0, 2.0))[None]
        result = ray_splat_depth_and_weight(
            transform, torch.tensor([[0.5, 0.5]], dtype=torch.float64), torch.tensor([0]), 1
        )
        torch.testing.assert_close(result.gaussian_weight, torch.ones(1, dtype=torch.float64))
        torch.testing.assert_close(result.depth, torch.tensor([2.0], dtype=torch.float64))
        self.assertTrue(bool(result.ray_plane_mask[0]))
        self.assertTrue(bool(result.valid_mask[0]))

    def test_ray_plane_and_low_pass_depth_branches_are_explicit(self) -> None:
        transform = transform_for_uv(0.5, 0.5, 0.2, -0.3, (1.0, 2.0, 3.0))[None]
        ray_plane = ray_splat_depth_and_weight(
            transform,
            torch.tensor([[2.0, 2.0]], dtype=torch.float64),
            torch.tensor([0]),
            1,
        )
        self.assertTrue(bool(ray_plane.ray_plane_mask[0]))
        torch.testing.assert_close(ray_plane.local_uv, torch.tensor([[0.2, -0.3]], dtype=torch.float64))
        torch.testing.assert_close(ray_plane.depth, torch.tensor([2.6], dtype=torch.float64))
        torch.testing.assert_close(
            ray_plane.gaussian_weight,
            torch.tensor([math.exp(-0.5 * (0.2**2 + 0.3**2))], dtype=torch.float64),
        )

        low_pass = ray_splat_depth_and_weight(
            transform,
            torch.tensor([[0.5, 0.5]], dtype=torch.float64),
            torch.tensor([0]),
            1,
        )
        self.assertFalse(bool(low_pass.ray_plane_mask[0]))
        torch.testing.assert_close(low_pass.depth, torch.tensor([3.0], dtype=torch.float64))
        torch.testing.assert_close(low_pass.gaussian_weight, torch.ones(1, dtype=torch.float64))

    def test_exact_and_configured_near_degeneracy_are_invalid(self) -> None:
        exact = torch.tensor(
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]],
            dtype=torch.float64,
        )
        exact_result = ray_splat_depth_and_weight(
            exact, torch.zeros((1, 2), dtype=torch.float64), torch.tensor([0]), 1
        )
        self.assertFalse(bool(exact_result.valid_mask[0]))
        torch.testing.assert_close(exact_result.depth, torch.zeros(1, dtype=torch.float64))

        near = transform_for_uv(0.5, 0.5, 0.0, 0.0, (0.0, 0.0, 2.0))[None]
        near = near.clone()
        near[0, 0] *= 1e-8
        near_result = ray_splat_depth_and_weight(
            near,
            torch.tensor([[0.5, 0.5]], dtype=torch.float64),
            torch.tensor([0]),
            1,
            hz_epsilon=1e-6,
        )
        self.assertFalse(bool(near_result.valid_mask[0]))

    def test_nonfinite_transform_is_invalid_without_nan_output(self) -> None:
        transform = transform_for_uv(0.5, 0.5, 0.0, 0.0, (0.0, 0.0, 2.0))[None]
        transform[0, 0, 0] = float("nan")
        result = ray_splat_depth_and_weight(
            transform,
            torch.tensor([[0.5, 0.5]], dtype=torch.float64),
            torch.tensor([0]),
            1,
        )
        self.assertFalse(bool(result.valid_mask[0]))
        self.assertTrue(bool(torch.isfinite(result.depth).all()))
        self.assertTrue(bool(torch.isfinite(result.gaussian_weight).all()))

    def test_transform_gradients_are_finite_away_from_branch_boundary(self) -> None:
        transform = transform_for_uv(0.5, 0.5, 0.2, -0.3, (1.0, 2.0, 3.0))[None]
        transform.requires_grad_(True)
        result = ray_splat_depth_and_weight(
            transform,
            torch.tensor([[2.0, 2.0]], dtype=torch.float64),
            torch.tensor([0]),
            1,
        )
        (result.depth + result.gaussian_weight).sum().backward()
        self.assertIsNotNone(transform.grad)
        self.assertTrue(bool(torch.isfinite(transform.grad).all()))
        self.assertGreater(float(transform.grad.abs().sum()), 0.0)

    def test_fixed_hit_ray_branch_passes_finite_difference_gradcheck(self) -> None:
        transform = transform_for_uv(0.5, 0.5, 0.2, -0.3, (1.0, 2.0, 3.0))[None]
        transform.requires_grad_(True)
        means2d = torch.tensor([[2.0, 2.0]], dtype=torch.float64, requires_grad=True)
        pixel_ids = torch.tensor([0])

        def outputs(current_transform: torch.Tensor, current_means: torch.Tensor):
            result = ray_splat_depth_and_weight(
                current_transform, current_means, pixel_ids, 1
            )
            return result.depth, result.gaussian_weight

        self.assertTrue(
            torch.autograd.gradcheck(
                outputs,
                (transform, means2d),
                eps=1e-6,
                atol=1e-5,
                rtol=1e-4,
            )
        )


if __name__ == "__main__":
    unittest.main()
