from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from unbiased_gsplat.corrected_depth import corrected_depth, segmented_cumulative_sum
from unbiased_gsplat.losses import (
    blending_weights,
    normal_consistency_from_rendered,
    normal_consistency_loss,
    segmented_exclusive_transmittance,
)


class SegmentedPropertyTests(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator().manual_seed(20260913)
        self.num_segments = 31
        self.counts = torch.randint(0, 7, (self.num_segments,), generator=generator)
        self.segment_ids = torch.repeat_interleave(
            torch.arange(self.num_segments), self.counts
        )
        self.generator = generator

    def test_cumulative_sum_matches_scalar_segment_loop(self) -> None:
        values = torch.rand(
            self.segment_ids.numel(), generator=self.generator, dtype=torch.float64
        )
        expected = torch.empty_like(values)
        running = 0.0
        previous_segment = -1
        for index, segment in enumerate(self.segment_ids.tolist()):
            if segment != previous_segment:
                running = 0.0
                previous_segment = segment
            running += float(values[index])
            expected[index] = running
        torch.testing.assert_close(
            segmented_cumulative_sum(values, self.segment_ids), expected
        )

    def test_exclusive_transmittance_matches_scalar_segment_loop(self) -> None:
        alpha = torch.rand(
            self.segment_ids.numel(), generator=self.generator, dtype=torch.float64
        ) * 0.7
        expected = torch.empty_like(alpha)
        running = 1.0
        previous_segment = -1
        for index, segment in enumerate(self.segment_ids.tolist()):
            if segment != previous_segment:
                running = 1.0
                previous_segment = segment
            expected[index] = running
            running *= 1.0 - float(alpha[index])
        torch.testing.assert_close(
            segmented_exclusive_transmittance(alpha, self.segment_ids),
            expected,
            atol=1e-12,
            rtol=1e-12,
        )

    def test_corrected_first_crossing_matches_scalar_loop_with_empty_segments(self) -> None:
        count = self.segment_ids.numel()
        depths = torch.linspace(1.0, 3.0, count, dtype=torch.float64)
        opacities = torch.rand(count, generator=self.generator, dtype=torch.float64) * 0.3
        gaussian_weights = torch.rand(count, generator=self.generator, dtype=torch.float64)
        valid = torch.rand(count, generator=self.generator) > 0.15
        actual = corrected_depth(
            depths,
            opacities,
            gaussian_weights,
            self.segment_ids,
            valid,
            num_segments=self.num_segments,
        )
        expected_depth = torch.zeros(self.num_segments, dtype=torch.float64)
        expected_valid = torch.zeros(self.num_segments, dtype=torch.bool)
        running = [0.0] * self.num_segments
        for index, segment in enumerate(self.segment_ids.tolist()):
            if not bool(valid[index]):
                continue
            running[segment] += float((opacities[index] + 0.1) * gaussian_weights[index])
            if not bool(expected_valid[segment]) and running[segment] > 0.6:
                expected_valid[segment] = True
                expected_depth[segment] = depths[index]
        torch.testing.assert_close(actual.valid_mask, expected_valid)
        torch.testing.assert_close(actual.depth, expected_depth)

    def test_dense_renderer_normal_identity_matches_expanded_hits_randomly(self) -> None:
        count = self.segment_ids.numel()
        raw_surfel_normals = torch.randn(
            (count, 3), generator=self.generator, dtype=torch.float64
        )
        surfel_normals = F.normalize(raw_surfel_normals, dim=-1)
        surface_normals = F.normalize(
            torch.randn(
                (self.num_segments, 3), generator=self.generator, dtype=torch.float64
            ),
            dim=-1,
        )
        surface_valid = torch.rand(
            self.num_segments, generator=self.generator
        ) > 0.2
        opacities = torch.rand(count, generator=self.generator, dtype=torch.float64) * 0.7
        gaussian_weights = torch.rand(count, generator=self.generator, dtype=torch.float64)
        hit_valid = torch.ones(count, dtype=torch.bool)
        expanded = normal_consistency_loss(
            surfel_normals,
            surface_normals,
            surface_valid,
            opacities,
            gaussian_weights,
            self.segment_ids,
            hit_valid,
            num_segments=self.num_segments,
            reduction="sum",
        )
        weights = blending_weights(
            opacities, gaussian_weights, self.segment_ids, hit_valid
        )
        rendered_alpha = torch.zeros(self.num_segments, dtype=torch.float64)
        rendered_alpha.scatter_add_(0, self.segment_ids, weights)
        rendered_normal = torch.zeros((self.num_segments, 3), dtype=torch.float64)
        rendered_normal.index_add_(0, self.segment_ids, weights[:, None] * surfel_normals)
        dense = normal_consistency_from_rendered(
            rendered_normal,
            rendered_alpha,
            surface_normals,
            surface_valid,
            reduction="sum",
        )
        torch.testing.assert_close(dense.loss, expanded.loss, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(
            dense.per_segment, expanded.per_segment, atol=1e-12, rtol=1e-12
        )


if __name__ == "__main__":
    unittest.main()
