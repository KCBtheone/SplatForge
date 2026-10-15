from __future__ import annotations

import unittest

import torch

from unbiased_gsplat.depth_normal import depth_to_normal


class CameraConventionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.depth = torch.full((1, 5, 5), 2.0, dtype=torch.float64)
        self.valid = torch.ones((1, 5, 5), dtype=torch.bool)
        self.intrinsics = torch.tensor(
            [[[100.0, 0.0, 2.5], [0.0, 100.0, 2.5], [0.0, 0.0, 1.0]]],
            dtype=torch.float64,
        )

    def test_front_plane_normal_is_oriented_toward_camera(self) -> None:
        camera_to_world = torch.eye(4, dtype=torch.float64)[None]
        result = depth_to_normal(
            self.depth, self.valid, camera_to_world, self.intrinsics
        )
        torch.testing.assert_close(
            result.normals[0, 2, 2], torch.tensor([0.0, 0.0, -1.0], dtype=torch.float64)
        )
        self.assertTrue(bool(result.valid_mask[0, 2, 2]))
        self.assertFalse(bool(result.valid_mask[0, 0, 0]))

    def test_world_rotation_is_applied_to_depth_normal(self) -> None:
        camera_to_world = torch.eye(4, dtype=torch.float64)[None]
        camera_to_world[0, :3, :3] = torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
            dtype=torch.float64,
        )
        camera_to_world[0, :3, 3] = torch.tensor([3.0, -2.0, 1.0], dtype=torch.float64)
        result = depth_to_normal(
            self.depth, self.valid, camera_to_world, self.intrinsics
        )
        torch.testing.assert_close(
            result.normals[0, 2, 2], torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float64)
        )

    def test_invalid_neighbor_invalidates_the_stencil(self) -> None:
        valid = self.valid.clone()
        valid[0, 2, 1] = False
        result = depth_to_normal(
            self.depth, valid, torch.eye(4, dtype=torch.float64)[None], self.intrinsics
        )
        self.assertFalse(bool(result.valid_mask[0, 2, 2]))
        torch.testing.assert_close(result.normals[0, 2, 2], torch.zeros(3, dtype=torch.float64))


if __name__ == "__main__":
    unittest.main()
