from __future__ import annotations

import unittest

import torch

from unbiased_gsplat.multiview import (
    PGSRMultiViewConfig,
    _sample_hwc,
    _unfold_fixed_patches,
    apply_homography,
    build_camera_neighbor_graph,
    multiview_consistency,
    normalized_cross_correlation,
    plane_induced_homography,
    weighted_multiview_regularization,
)


def camera(translation_x: float = 0.0, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    result = torch.eye(4, dtype=dtype)
    result[0, 3] = translation_x
    return result


def intrinsics(height: int, width: int, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.tensor(
        [
            [20.0, 0.0, width / 2.0],
            [0.0, 20.0, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=dtype,
    )


def plane_maps(
    height: int,
    width: int,
    camera_to_world: torch.Tensor,
    K: torch.Tensor,
    *,
    depth: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    y, x = torch.meshgrid(
        torch.arange(height, dtype=K.dtype),
        torch.arange(width, dtype=K.dtype),
        indexing="ij",
    )
    pixels = torch.stack((x + 0.5, y + 0.5, torch.ones_like(x)), dim=-1)
    directions = pixels @ torch.linalg.inv(K).T
    points_camera = directions * depth
    points_world = (
        points_camera @ camera_to_world[:3, :3].T
        + camera_to_world[:3, 3]
    )
    normals_world = torch.tensor(
        [0.0, 0.0, -1.0], dtype=K.dtype
    ).expand_as(points_world).clone()
    texture = 0.45 + 0.7 * points_world[..., 0] + 0.35 * points_world[..., 1]
    image = torch.stack(
        (texture, 0.1 + 0.8 * texture, 0.2 + 0.6 * texture), dim=-1
    )
    return points_world, normals_world, image


class MultiViewTests(unittest.TestCase):
    def test_plane_homography_matches_direct_projection(self) -> None:
        height = width = 9
        K = intrinsics(height, width)
        source_camera = camera(0.0)
        target_camera = camera(0.1)
        source_points, normals, _ = plane_maps(
            height, width, source_camera, K, depth=2.0
        )
        pixels = torch.tensor(
            [[2.5, 3.5], [4.5, 4.5], [6.5, 5.5]], dtype=torch.float64
        )
        integer_x = (pixels[:, 0] - 0.5).long()
        integer_y = (pixels[:, 1] - 0.5).long()
        points = source_points[integer_y, integer_x]
        selected_normals = normals[integer_y, integer_x]
        homography, valid = plane_induced_homography(
            source_camera,
            target_camera,
            K,
            K,
            points,
            selected_normals,
        )
        mapped, mapped_valid = apply_homography(homography, pixels)

        target_world_to_camera = torch.linalg.inv(target_camera)
        target_points = points @ target_world_to_camera[:3, :3].T
        target_points = target_points + target_world_to_camera[:3, 3]
        direct_homogeneous = target_points @ K.T
        direct = direct_homogeneous[:, :2] / direct_homogeneous[:, 2:]
        self.assertTrue(bool(valid.all()))
        self.assertTrue(bool(mapped_valid.all()))
        torch.testing.assert_close(mapped, direct, atol=1e-12, rtol=1e-12)

    def test_neighbor_graph_uses_angle_then_position_and_limits_count(self) -> None:
        cameras = torch.eye(4, dtype=torch.float64).repeat(4, 1, 1)
        cameras[:, 0, 3] = torch.tensor([0.0, 0.02, 0.2, 2.0])
        graph = build_camera_neighbor_graph(
            cameras,
            [10, 11, 12, 13],
            config=PGSRMultiViewConfig(
                max_neighbors=2,
                min_relative_position=0.01,
                max_relative_position=1.5,
            ),
        )
        self.assertEqual(graph[0], (1, 2))
        self.assertNotIn(3, graph[0])

    def test_ncc_is_affine_brightness_invariant_and_rejects_constant_patch(self) -> None:
        reference = torch.tensor(
            [[0.0, 0.5, 1.0], [2.0, 2.0, 2.0]], dtype=torch.float64
        )
        neighbor = torch.tensor(
            [[3.0, 4.0, 5.0], [7.0, 7.0, 7.0]], dtype=torch.float64
        )
        correlation, valid = normalized_cross_correlation(reference, neighbor)
        torch.testing.assert_close(correlation[0], torch.tensor(1.0, dtype=torch.float64))
        self.assertTrue(bool(valid[0]))
        self.assertFalse(bool(valid[1]))
        torch.testing.assert_close(correlation[1], torch.tensor(0.0, dtype=torch.float64))

    def test_consistent_plane_pair_has_zero_reprojection_and_unit_ncc(self) -> None:
        height = width = 11
        K = intrinsics(height, width)
        reference_camera = camera(0.0)
        neighbor_camera = camera(0.1)
        reference_points, reference_normals, reference_image = plane_maps(
            height, width, reference_camera, K, depth=2.0
        )
        neighbor_points, neighbor_normals, neighbor_image = plane_maps(
            height, width, neighbor_camera, K, depth=2.0
        )
        reference_points = reference_points.clone().requires_grad_(True)
        neighbor_points = neighbor_points.clone().requires_grad_(True)
        valid = torch.ones((height, width), dtype=torch.bool)
        config = PGSRMultiViewConfig(patch_size=3, pixel_stride=2)
        result = multiview_consistency(
            reference_image,
            neighbor_image,
            reference_points,
            reference_normals,
            valid,
            neighbor_points,
            neighbor_normals,
            valid,
            valid,
            valid,
            reference_camera,
            neighbor_camera,
            K,
            K,
            config=config,
        )
        self.assertGreater(int(result.geometry_denominator.item()), 0)
        self.assertGreater(int(result.photometric_denominator.item()), 0)
        torch.testing.assert_close(
            result.geometry_loss, torch.tensor(0.0, dtype=torch.float64), atol=1e-11, rtol=0.0
        )
        torch.testing.assert_close(
            result.photometric_loss, torch.tensor(0.0, dtype=torch.float64), atol=1e-11, rtol=0.0
        )
        loss = weighted_multiview_regularization(result, config)
        loss.backward()
        self.assertIsNotNone(reference_points.grad)
        self.assertIsNotNone(neighbor_points.grad)
        self.assertTrue(bool(torch.isfinite(reference_points.grad).all()))
        self.assertTrue(bool(torch.isfinite(neighbor_points.grad).all()))

    def test_inconsistent_reverse_plane_is_removed_by_occlusion_gate(self) -> None:
        height = width = 11
        K = intrinsics(height, width)
        reference_camera = camera(0.0)
        neighbor_camera = camera(0.1)
        reference_points, reference_normals, reference_image = plane_maps(
            height, width, reference_camera, K, depth=2.0
        )
        neighbor_points, neighbor_normals, neighbor_image = plane_maps(
            height, width, neighbor_camera, K, depth=4.0
        )
        valid = torch.ones((height, width), dtype=torch.bool)
        result = multiview_consistency(
            reference_image,
            neighbor_image,
            reference_points,
            reference_normals,
            valid,
            neighbor_points,
            neighbor_normals,
            valid,
            valid,
            valid,
            reference_camera,
            neighbor_camera,
            K,
            K,
            config=PGSRMultiViewConfig(
                patch_size=3,
                pixel_stride=2,
                reprojection_threshold_pixels=0.1,
            ),
        )
        self.assertEqual(int(result.geometry_denominator.item()), 0)
        self.assertEqual(int(result.photometric_denominator.item()), 0)
        torch.testing.assert_close(result.geometry_loss, torch.tensor(0.0, dtype=torch.float64))
        torch.testing.assert_close(result.photometric_loss, torch.tensor(0.0, dtype=torch.float64))

    def test_unfold_matches_grid_sample_at_integer_centers(self) -> None:
        height, width = 13, 17
        dtype = torch.float64
        image = torch.arange(height * width, dtype=dtype).reshape(height, width, 1)
        image = image / float(height * width)
        for pixel_stride in (1, 2, 4):
            radius = 1
            ys = torch.arange(radius, height - radius, pixel_stride)
            xs = torch.arange(radius, width - radius, pixel_stride)
            y_grid, x_grid = torch.meshgrid(ys, xs, indexing="ij")
            centers = torch.stack(
                (
                    x_grid.reshape(-1).to(dtype) + 0.5,
                    y_grid.reshape(-1).to(dtype) + 0.5,
                ),
                dim=-1,
            )
            offsets_y, offsets_x = torch.meshgrid(
                torch.arange(-radius, radius + 1),
                torch.arange(-radius, radius + 1),
                indexing="ij",
            )
            offsets = torch.stack((offsets_x, offsets_y), dim=-1).reshape(-1, 2).to(dtype)
            grid = centers[:, None, :] + offsets[None]
            expected = _sample_hwc(image, grid)[..., 0]
            actual = _unfold_fixed_patches(
                image,
                patch_size=3,
                pixel_stride=pixel_stride,
                expected_rows=int(centers.shape[0]),
            )
            torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0.0)

    def test_unfold_rejects_layout_mismatch(self) -> None:
        image = torch.zeros(11, 11, 1, dtype=torch.float64)
        with self.assertRaises(ValueError):
            _unfold_fixed_patches(
                image, patch_size=3, pixel_stride=2, expected_rows=3
            )


if __name__ == "__main__":
    unittest.main()
