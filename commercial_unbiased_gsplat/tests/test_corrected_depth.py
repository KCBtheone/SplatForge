from __future__ import annotations

import unittest

import torch

from unbiased_gsplat.corrected_depth import corrected_depth, segmented_cumulative_sum


class CorrectedDepthTests(unittest.TestCase):
    def test_segmented_cumulative_sum_resets(self) -> None:
        actual = segmented_cumulative_sum(
            torch.tensor([0.1, 0.2, 1.0, 2.0], dtype=torch.float64),
            torch.tensor([0, 0, 2, 2]),
        )
        torch.testing.assert_close(actual, torch.tensor([0.1, 0.3, 1.0, 3.0], dtype=torch.float64))

    def test_three_low_opacity_hits_jointly_cross_default_threshold(self) -> None:
        result = corrected_depth(
            torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
            torch.full((3,), 0.11, dtype=torch.float64),
            torch.ones(3, dtype=torch.float64),
            torch.zeros(3, dtype=torch.int64),
            torch.ones(3, dtype=torch.bool),
            num_segments=2,
        )
        torch.testing.assert_close(result.cumulative_contribution, torch.tensor([0.21, 0.42, 0.63], dtype=torch.float64))
        torch.testing.assert_close(result.depth, torch.tensor([3.0, 0.0], dtype=torch.float64))
        torch.testing.assert_close(result.valid_mask, torch.tensor([True, False]))
        torch.testing.assert_close(result.selected_hit_indices, torch.tensor([2, -1]))

    def test_threshold_comparison_is_strict(self) -> None:
        result = corrected_depth(
            torch.tensor([1.0, 2.0], dtype=torch.float64),
            torch.tensor([0.25, 0.25], dtype=torch.float64),
            torch.ones(2, dtype=torch.float64),
            torch.zeros(2, dtype=torch.int64),
            torch.ones(2, dtype=torch.bool),
            num_segments=1,
            opacity_epsilon=0.0,
            threshold=0.5,
        )
        self.assertFalse(bool(result.valid_mask[0]))
        self.assertEqual(int(result.selected_hit_indices[0]), -1)

    def test_first_hit_can_cross_and_local_depth_inversion_is_not_resorted(self) -> None:
        result = corrected_depth(
            torch.tensor([5.0, 1.0], dtype=torch.float64),
            torch.tensor([0.7, 0.7], dtype=torch.float64),
            torch.ones(2, dtype=torch.float64),
            torch.zeros(2, dtype=torch.int64),
            torch.ones(2, dtype=torch.bool),
            num_segments=1,
        )
        torch.testing.assert_close(result.depth, torch.tensor([5.0], dtype=torch.float64))
        self.assertEqual(int(result.selected_hit_indices[0]), 0)

    def test_invalid_hits_contribute_nothing(self) -> None:
        result = corrected_depth(
            torch.tensor([9.0, 2.0], dtype=torch.float64),
            torch.tensor([1.0, 0.7], dtype=torch.float64),
            torch.ones(2, dtype=torch.float64),
            torch.zeros(2, dtype=torch.int64),
            torch.tensor([False, True]),
            num_segments=1,
        )
        torch.testing.assert_close(result.depth, torch.tensor([2.0], dtype=torch.float64))
        self.assertEqual(int(result.selected_hit_indices[0]), 1)

    def test_gradient_flows_only_to_selected_depth(self) -> None:
        depths = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
        result = corrected_depth(
            depths,
            torch.tensor([0.125, 0.125, 0.125], dtype=torch.float64),
            torch.ones(3, dtype=torch.float64),
            torch.zeros(3, dtype=torch.int64),
            torch.ones(3, dtype=torch.bool),
            num_segments=1,
        )
        result.depth.sum().backward()
        torch.testing.assert_close(depths.grad, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))

    def test_first_crossing_per_segment_uses_broadcast_sentinels(self) -> None:
        depths = torch.tensor([1.0, 2.0, 9.0, 4.0], dtype=torch.float64, requires_grad=True)
        result = corrected_depth(
            depths,
            torch.tensor([0.7, 0.7, 0.7, 0.7], dtype=torch.float64),
            torch.ones(4, dtype=torch.float64),
            torch.tensor([0, 0, 1, 1]),
            torch.ones(4, dtype=torch.bool),
            num_segments=3,
        )
        torch.testing.assert_close(result.depth, torch.tensor([1.0, 9.0, 0.0], dtype=torch.float64))
        torch.testing.assert_close(result.valid_mask, torch.tensor([True, True, False]))
        torch.testing.assert_close(result.selected_hit_indices, torch.tensor([0, 2, -1]))
        result.depth.sum().backward()
        torch.testing.assert_close(
            depths.grad, torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float64)
        )

    def test_empty_hit_list_produces_explicit_invalid_dense_output(self) -> None:
        empty = torch.empty(0, dtype=torch.float64)
        result = corrected_depth(
            empty,
            empty,
            empty,
            torch.empty(0, dtype=torch.int64),
            torch.empty(0, dtype=torch.bool),
            num_segments=3,
        )
        torch.testing.assert_close(result.depth, torch.zeros(3, dtype=torch.float64))
        torch.testing.assert_close(result.valid_mask, torch.zeros(3, dtype=torch.bool))
        torch.testing.assert_close(result.selected_hit_indices, torch.full((3,), -1, dtype=torch.int64))


if __name__ == "__main__":
    unittest.main()
