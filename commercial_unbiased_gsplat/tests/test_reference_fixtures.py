from __future__ import annotations

import unittest
from pathlib import Path

import torch

from unbiased_gsplat.convergence import depth_convergence
from unbiased_gsplat.corrected_depth import corrected_depth
from unbiased_gsplat.ray_splat_depth import ray_splat_depth_and_weight


FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "reference_fixtures"
FIXTURES = sorted(FIXTURE_ROOT.glob("*.pt"))


@unittest.skipUnless(
    FIXTURES,
    "BLOCKED: Track R has not supplied source-free black-box tensor fixtures",
)
class ReferenceFixtureTests(unittest.TestCase):
    def test_all_track_r_fixtures(self) -> None:
        for path in FIXTURES:
            with self.subTest(fixture=path.name):
                fixture = torch.load(path, map_location="cpu", weights_only=True)
                samples = ray_splat_depth_and_weight(
                    fixture["ray_transforms"],
                    fixture["means2d"],
                    fixture["pixel_ids"],
                    int(fixture["image_width"]),
                    hz_epsilon=float(fixture.get("hz_epsilon", 0.0)),
                )
                torch.testing.assert_close(samples.gaussian_weight, fixture["expected_gaussian_weight"])
                torch.testing.assert_close(samples.depth, fixture["expected_ray_depth"])
                torch.testing.assert_close(samples.valid_mask, fixture["expected_hit_valid"])
                corrected = corrected_depth(
                    samples.depth,
                    fixture["opacities"],
                    samples.gaussian_weight,
                    fixture["segment_ids"],
                    samples.valid_mask,
                    num_segments=int(fixture["num_segments"]),
                    opacity_epsilon=float(fixture.get("opacity_epsilon", 0.1)),
                    threshold=float(fixture.get("threshold", 0.6)),
                )
                torch.testing.assert_close(corrected.valid_mask, fixture["expected_corrected_valid"])
                torch.testing.assert_close(corrected.depth, fixture["expected_corrected_depth"])
                convergence = depth_convergence(
                    samples.depth,
                    samples.gaussian_weight,
                    fixture["segment_ids"],
                    samples.valid_mask,
                    scene_radius=float(fixture["scene_radius"]),
                    num_segments=int(fixture["num_segments"]),
                    reduction=str(fixture.get("convergence_reduction", "sum")),
                    far_gradient_scale=float(fixture.get("far_gradient_scale", 1.25)),
                )
                torch.testing.assert_close(convergence.numerator, fixture["expected_convergence_numerator"])


if __name__ == "__main__":
    unittest.main()
