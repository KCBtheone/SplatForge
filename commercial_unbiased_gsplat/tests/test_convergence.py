from __future__ import annotations

import unittest

import torch

from unbiased_gsplat.convergence import depth_convergence, segmented_adjacent_pairs


class ConvergenceForwardTests(unittest.TestCase):
    def test_pairs_never_cross_pixel_segments(self) -> None:
        previous, current = segmented_adjacent_pairs(torch.tensor([0, 0, 1, 1, 1, 4]))
        torch.testing.assert_close(previous, torch.tensor([0, 2, 3]))
        torch.testing.assert_close(current, torch.tensor([1, 3, 4]))

    def test_weight_detach_threshold_and_sum(self) -> None:
        depths = torch.tensor([0.0, 1.0, 2.0000001], dtype=torch.float64)
        weights = torch.tensor([0.8, 0.5, 0.4], dtype=torch.float64)
        result = depth_convergence(
            depths,
            weights,
            torch.tensor([0, 0, 0]),
            torch.ones(3, dtype=torch.bool),
            scene_radius=4.0,
            num_segments=2,
            reduction="sum",
        )
        torch.testing.assert_close(result.pair_mask, torch.tensor([True, False]))
        torch.testing.assert_close(result.numerator, torch.tensor(0.5, dtype=torch.float64))
        torch.testing.assert_close(result.per_segment, torch.tensor([0.5, 0.0], dtype=torch.float64))
        torch.testing.assert_close(result.denominator, torch.tensor(1.0, dtype=torch.float64))

    def test_difference_exactly_at_radius_quarter_is_included(self) -> None:
        result = depth_convergence(
            torch.tensor([2.0, 3.0], dtype=torch.float64),
            torch.tensor([0.25, 0.75], dtype=torch.float64),
            torch.tensor([0, 0]),
            torch.ones(2, dtype=torch.bool),
            scene_radius=4.0,
            num_segments=1,
            reduction="mean_pairs",
        )
        self.assertTrue(bool(result.pair_mask[0]))
        torch.testing.assert_close(result.loss, torch.tensor(0.25, dtype=torch.float64))
        torch.testing.assert_close(result.denominator, torch.tensor(1.0, dtype=torch.float64))

    def test_no_pairs_retains_a_differentiable_zero(self) -> None:
        depth = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
        result = depth_convergence(
            depth,
            torch.ones_like(depth),
            torch.tensor([0]),
            torch.ones(1, dtype=torch.bool),
            scene_radius=1.0,
            num_segments=4,
            reduction="mean_segments",
        )
        result.loss.backward()
        torch.testing.assert_close(depth.grad, torch.zeros_like(depth))
        torch.testing.assert_close(result.denominator, torch.tensor(4.0, dtype=torch.float64))


class ConvergenceBackwardTests(unittest.TestCase):
    def test_far_hit_gradient_is_scaled_and_weight_has_no_gradient(self) -> None:
        depths = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
        weights = torch.tensor([0.5, 0.75], dtype=torch.float64, requires_grad=True)
        result = depth_convergence(
            depths,
            weights,
            torch.tensor([0, 0]),
            torch.ones(2, dtype=torch.bool),
            scene_radius=100.0,
            num_segments=1,
            reduction="sum",
            far_gradient_scale=1.25,
        )
        result.loss.backward()
        torch.testing.assert_close(depths.grad, torch.tensor([-1.0, 1.25], dtype=torch.float64))
        self.assertIsNone(weights.grad)

    def test_segment_order_defines_the_scaled_current_hit_even_if_depth_inverts(self) -> None:
        depths = torch.tensor([3.0, 2.0], dtype=torch.float64, requires_grad=True)
        result = depth_convergence(
            depths,
            torch.ones(2, dtype=torch.float64),
            torch.tensor([0, 0]),
            torch.ones(2, dtype=torch.bool),
            scene_radius=100.0,
            num_segments=1,
            reduction="sum",
        )
        result.loss.backward()
        torch.testing.assert_close(depths.grad, torch.tensor([2.0, -2.5], dtype=torch.float64))

    def test_gradient_descent_moves_a_deeper_current_hit_toward_camera(self) -> None:
        depths = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
        before_gap = float((depths[1] - depths[0]).detach())
        result = depth_convergence(
            depths,
            torch.ones(2, dtype=torch.float64),
            torch.tensor([0, 0]),
            torch.ones(2, dtype=torch.bool),
            scene_radius=100.0,
            num_segments=1,
            reduction="sum",
        )
        result.loss.backward()
        with torch.no_grad():
            updated = depths - 0.1 * depths.grad
        self.assertLess(float(updated[1].detach()), float(depths[1].detach()))
        self.assertLess(float((updated[1] - updated[0]).detach()), before_gap)


if __name__ == "__main__":
    unittest.main()
