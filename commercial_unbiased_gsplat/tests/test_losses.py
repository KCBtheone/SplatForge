from __future__ import annotations

import unittest

import torch

from unbiased_gsplat.losses import (
    blending_weights,
    normal_consistency_loss,
    normal_consistency_from_rendered,
    segmented_exclusive_transmittance,
)


class LossTests(unittest.TestCase):
    def test_segmented_transmittance_resets_per_pixel(self) -> None:
        alpha = torch.tensor([0.5, 0.5, 0.25, 0.5], dtype=torch.float64)
        segment_ids = torch.tensor([0, 0, 1, 1])
        torch.testing.assert_close(
            segmented_exclusive_transmittance(alpha, segment_ids),
            torch.tensor([1.0, 0.5, 1.0, 0.75], dtype=torch.float64),
        )

    def test_blending_weight_matches_front_to_back_formula(self) -> None:
        actual = blending_weights(
            torch.tensor([0.5, 0.5], dtype=torch.float64),
            torch.ones(2, dtype=torch.float64),
            torch.tensor([0, 0]),
            torch.ones(2, dtype=torch.bool),
        )
        torch.testing.assert_close(actual, torch.tensor([0.5, 0.25], dtype=torch.float64))

    def test_normal_loss_uses_each_hits_blending_weight(self) -> None:
        result = normal_consistency_loss(
            torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]], dtype=torch.float64),
            torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float64),
            torch.tensor([True]),
            torch.tensor([0.5, 0.5], dtype=torch.float64),
            torch.ones(2, dtype=torch.float64),
            torch.tensor([0, 0]),
            torch.ones(2, dtype=torch.bool),
            num_segments=1,
            reduction="sum",
        )
        torch.testing.assert_close(result.blending_weights, torch.tensor([0.5, 0.25], dtype=torch.float64))
        torch.testing.assert_close(result.per_hit, torch.tensor([0.0, 0.5], dtype=torch.float64))
        torch.testing.assert_close(result.loss, torch.tensor(0.5, dtype=torch.float64))

    def test_dense_renderer_identity_matches_expanded_hit_sum(self) -> None:
        dense = normal_consistency_from_rendered(
            torch.tensor([[[[0.0, 0.0, -0.25]]]], dtype=torch.float64),
            torch.tensor([[[[0.75]]]], dtype=torch.float64),
            torch.tensor([[[[0.0, 0.0, -1.0]]]], dtype=torch.float64),
            torch.tensor([[[True]]]),
            reduction="sum",
        )
        # 0.5 * (1 - 1) + 0.25 * (1 - (-1)) = 0.5.
        torch.testing.assert_close(dense.loss, torch.tensor(0.5, dtype=torch.float64))


if __name__ == "__main__":
    unittest.main()
