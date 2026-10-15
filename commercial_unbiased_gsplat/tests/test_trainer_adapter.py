from __future__ import annotations

import unittest

import torch

from unbiased_gsplat.hit_sequence import make_hit_sequence
from unbiased_gsplat.trainer_adapter import (
    clamp_log_tangent_scales_after_densification_,
    UnbiasedSurfelConfig,
    clamp_tangent_scales_after_densification_,
    compute_corrected_surface_chunked_from_hit_sequence,
    compute_corrected_surface_from_hit_sequence,
    compute_from_hit_sequence,
    densification_enabled,
    prepare_color_target,
    weighted_regularization,
)


def plane_fixture() -> tuple[dict[str, torch.Tensor | int | None], object]:
    height = width = 5
    num_gaussians = height * width
    pixel_ids = torch.arange(num_gaussians, dtype=torch.int64)
    pixel_x = (pixel_ids % width).to(torch.float64) + 0.5
    pixel_y = (pixel_ids // width).to(torch.float64) + 0.5
    depth = 2.0
    transforms = torch.zeros((1, num_gaussians, 3, 3), dtype=torch.float64)
    transforms[0, :, 0, 0] = 1.0
    transforms[0, :, 0, 2] = pixel_x * depth
    transforms[0, :, 1, 1] = 1.0
    transforms[0, :, 1, 2] = pixel_y * depth
    transforms[0, :, 2, 2] = depth
    meta = {
        "means2d": torch.stack((pixel_x, pixel_y), dim=-1)[None],
        "ray_transforms": transforms,
        "opacities": torch.full((1, num_gaussians), 0.9, dtype=torch.float64),
        "depths": torch.full((1, num_gaussians), depth, dtype=torch.float64),
        "normals": torch.tensor([0.0, 0.0, -1.0], dtype=torch.float64).expand(1, num_gaussians, 3).clone(),
    }
    sequence = make_hit_sequence(
        torch.arange(num_gaussians),
        pixel_ids,
        torch.zeros(num_gaussians, dtype=torch.int64),
        num_images=1,
        image_height=height,
        image_width=width,
    )
    return meta, sequence


class TrainerAdapterTests(unittest.TestCase):
    def test_compact_multiview_surface_matches_full_adapter(self) -> None:
        meta, sequence = plane_fixture()
        viewmats = torch.eye(4, dtype=torch.float64)[None]
        intrinsics = torch.eye(3, dtype=torch.float64)[None]
        full = compute_from_hit_sequence(
            meta,
            sequence,
            viewmats,
            intrinsics,
            scene_radius=10.0,
        )
        compact = compute_corrected_surface_from_hit_sequence(
            meta,
            sequence,
            viewmats,
            intrinsics,
        )
        torch.testing.assert_close(compact.points, full.depth_normal.points)
        self.assertTrue(torch.equal(compact.valid_mask, full.corrected_depth.valid_mask.reshape(1, 5, 5)))
        self.assertEqual(compact.num_hits, sequence.num_hits)
        self.assertEqual(compact.corrected_valid_pixels, 25)

    def test_chunked_surface_preserves_crossing_across_chunk_boundaries(self) -> None:
        base_meta, _ = plane_fixture()
        repeats = 3
        meta = {
            name: value.repeat_interleave(repeats, dim=1)
            for name, value in base_meta.items()
            if isinstance(value, torch.Tensor)
        }
        meta["opacities"] = torch.full_like(meta["opacities"], 0.15)
        pixel_ids = torch.arange(25, dtype=torch.int64).repeat_interleave(repeats)
        sequence = make_hit_sequence(
            torch.arange(pixel_ids.numel(), dtype=torch.int64),
            pixel_ids,
            torch.zeros_like(pixel_ids),
            num_images=1,
            image_height=5,
            image_width=5,
        )
        viewmats = torch.eye(4, dtype=torch.float64)[None]
        intrinsics = torch.eye(3, dtype=torch.float64)[None]
        full = compute_from_hit_sequence(
            meta,
            sequence,
            viewmats,
            intrinsics,
            scene_radius=10.0,
        )
        chunked = compute_corrected_surface_chunked_from_hit_sequence(
            meta,
            sequence,
            viewmats,
            intrinsics,
            max_hits_per_chunk=2,
        )
        torch.testing.assert_close(chunked.points, full.depth_normal.points)
        self.assertTrue(
            torch.equal(
                chunked.valid_mask,
                full.corrected_depth.valid_mask.reshape(1, 5, 5),
            )
        )
        self.assertEqual(chunked.num_hits, 75)
        self.assertEqual(chunked.corrected_valid_pixels, 25)

    def test_end_to_end_tensor_adapter_on_front_plane(self) -> None:
        meta, sequence = plane_fixture()
        viewmats = torch.eye(4, dtype=torch.float64)[None]
        intrinsics = torch.eye(3, dtype=torch.float64)[None]
        result = compute_from_hit_sequence(
            meta,
            sequence,
            viewmats,
            intrinsics,
            scene_radius=10.0,
        )
        self.assertTrue(bool(result.corrected_depth.valid_mask.all()))
        torch.testing.assert_close(
            result.corrected_depth.depth, torch.full((25,), 2.0, dtype=torch.float64)
        )
        torch.testing.assert_close(result.convergence.loss, torch.tensor(0.0, dtype=torch.float64))
        torch.testing.assert_close(result.normal_consistency.loss, torch.tensor(0.0, dtype=torch.float64))
        torch.testing.assert_close(weighted_regularization(result), torch.tensor(0.0, dtype=torch.float64))

    def test_primary_renderer_outputs_take_exact_dense_normal_path(self) -> None:
        meta, sequence = plane_fixture()
        render_alphas = torch.full((1, 5, 5, 1), 0.9, dtype=torch.float64)
        render_normals = torch.tensor([0.0, 0.0, -0.9], dtype=torch.float64).expand(1, 5, 5, 3).clone()
        result = compute_from_hit_sequence(
            meta,
            sequence,
            torch.eye(4, dtype=torch.float64)[None],
            torch.eye(3, dtype=torch.float64)[None],
            scene_radius=10.0,
            rendered_alphas=render_alphas,
            rendered_normals=render_normals,
        )
        torch.testing.assert_close(result.normal_consistency.loss, torch.tensor(0.0, dtype=torch.float64))

    def test_primary_renderer_world_normals_are_not_rotated_again(self) -> None:
        meta, sequence = plane_fixture()
        render_alphas = torch.full((1, 5, 5, 1), 0.9, dtype=torch.float64)
        angle = torch.tensor(torch.pi / 2.0, dtype=torch.float64)
        cosine = torch.cos(angle)
        sine = torch.sin(angle)
        camera_to_world = torch.tensor(
            [
                [cosine, 0.0, sine, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-sine, 0.0, cosine, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )[None]
        viewmat = torch.linalg.inv(camera_to_world)
        expected_world_normal = camera_to_world[0, :3, :3] @ torch.tensor(
            [0.0, 0.0, -1.0], dtype=torch.float64
        )
        render_normals_world = (0.9 * expected_world_normal).expand(
            1, 5, 5, 3
        ).clone()
        result = compute_from_hit_sequence(
            meta,
            sequence,
            viewmat,
            torch.eye(3, dtype=torch.float64)[None],
            scene_radius=10.0,
            rendered_alphas=render_alphas,
            rendered_normals=render_normals_world,
        )
        torch.testing.assert_close(
            result.normal_consistency.loss, torch.tensor(0.0, dtype=torch.float64)
        )

    def test_meta_camera_normals_are_rotated_to_world(self) -> None:
        meta, sequence = plane_fixture()
        angle = torch.tensor(torch.pi / 2.0, dtype=torch.float64)
        cosine = torch.cos(angle)
        sine = torch.sin(angle)
        camera_to_world = torch.tensor(
            [
                [cosine, 0.0, sine, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-sine, 0.0, cosine, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )[None]
        result = compute_from_hit_sequence(
            meta,
            sequence,
            torch.linalg.inv(camera_to_world),
            torch.eye(3, dtype=torch.float64)[None],
            scene_radius=10.0,
        )
        torch.testing.assert_close(
            result.normal_consistency.loss, torch.tensor(0.0, dtype=torch.float64)
        )

    def test_dtu_gamma_is_not_applied_to_other_datasets(self) -> None:
        target = torch.tensor([0.25], dtype=torch.float64)
        torch.testing.assert_close(
            prepare_color_target(target, dataset_kind="DTU"),
            torch.tensor([0.5], dtype=torch.float64),
        )
        self.assertIs(prepare_color_target(target, dataset_kind="mipnerf360"), target)

    def test_densification_stop_and_explicit_tangent_scale_clamp(self) -> None:
        config = UnbiasedSurfelConfig(densification_stop_iteration=20_000)
        self.assertTrue(densification_enabled(19_999, config))
        self.assertFalse(densification_enabled(20_000, config))
        scales = torch.tensor([[2.0, 3.0, 9.0]])
        clamp_tangent_scales_after_densification_(
            scales, 1.5, iteration=20_000, config=config
        )
        torch.testing.assert_close(scales, torch.tensor([[1.5, 1.5, 9.0]]))

        log_scales = torch.log(torch.tensor([[2.0, 3.0, 9.0]]))
        clamp_log_tangent_scales_after_densification_(
            log_scales, 1.5, iteration=20_000, config=config
        )
        torch.testing.assert_close(
            torch.exp(log_scales), torch.tensor([[1.5, 1.5, 9.0]])
        )

    def test_log_scale_clamp_rejects_nonpositive_physical_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            clamp_log_tangent_scales_after_densification_(
                torch.zeros((1, 3)), 0.0, iteration=20_000
            )


if __name__ == "__main__":
    unittest.main()
