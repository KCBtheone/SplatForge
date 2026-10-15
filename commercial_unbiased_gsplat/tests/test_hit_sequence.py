from __future__ import annotations

import unittest
import sys
import types
from unittest.mock import patch

import torch

from unbiased_gsplat.hit_sequence import (
    UnsupportedGsplatLayoutError,
    center_depth_order_violations,
    collect_ordered_hits,
    gather_hit_attributes,
    make_hit_sequence,
)


class HitSequenceTests(unittest.TestCase):
    def test_dense_segment_mapping_and_gather(self) -> None:
        sequence = make_hit_sequence(
            torch.tensor([1, 0, 1]),
            torch.tensor([0, 3, 3]),
            torch.tensor([0, 1, 1]),
            num_images=2,
            image_height=2,
            image_width=2,
        )
        torch.testing.assert_close(sequence.segment_ids, torch.tensor([0, 7, 7]))
        meta = {
            "means2d": torch.arange(8, dtype=torch.float64).reshape(2, 2, 2),
            "ray_transforms": torch.arange(36, dtype=torch.float64).reshape(2, 2, 3, 3),
            "opacities": torch.arange(4, dtype=torch.float64).reshape(2, 2),
            "depths": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64),
            "normals": torch.arange(12, dtype=torch.float64).reshape(2, 2, 3),
        }
        gathered = gather_hit_attributes(meta, sequence)
        torch.testing.assert_close(gathered.center_depths, torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64))

    def test_out_of_order_segments_are_rejected_not_sorted(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be nondecreasing"):
            make_hit_sequence(
                torch.tensor([0, 0]),
                torch.tensor([1, 0]),
                torch.tensor([0, 0]),
                num_images=1,
                image_height=1,
                image_width=2,
            )

    def test_center_depth_violations_report_current_indices(self) -> None:
        sequence = make_hit_sequence(
            torch.tensor([0, 1, 2]),
            torch.tensor([0, 0, 0]),
            torch.tensor([0, 0, 0]),
            num_images=1,
            image_height=1,
            image_width=1,
        )
        actual = center_depth_order_violations(
            torch.tensor([1.0, 3.0, 2.0]), sequence
        )
        torch.testing.assert_close(actual, torch.tensor([2]))

    def test_packed_metadata_is_rejected_before_importing_gsplat(self) -> None:
        packed_meta = {
            "means2d": torch.zeros((2, 2)),
            "ray_transforms": torch.zeros((2, 3, 3)),
            "opacities": torch.zeros(2),
            "isect_offsets": torch.zeros((1, 1), dtype=torch.int32),
            "flatten_ids": torch.zeros(0, dtype=torch.int32),
            "width": 1,
            "height": 1,
            "tile_size": 16,
            "camera_ids": torch.zeros(2, dtype=torch.int64),
            "gaussian_ids": torch.zeros(2, dtype=torch.int64),
        }
        with self.assertRaises(UnsupportedGsplatLayoutError):
            collect_ordered_hits(packed_meta)

    def test_unpacked_adapter_preserves_indices_returned_by_gsplat(self) -> None:
        captured = {}

        def fake_indices(*args):
            captured["args"] = args
            return (
                torch.tensor([1, 0, 2]),
                torch.tensor([0, 3, 3]),
                torch.tensor([0, 1, 1]),
            )

        gsplat_module = types.ModuleType("gsplat")
        cuda_module = types.ModuleType("gsplat.cuda")
        wrapper_module = types.ModuleType("gsplat.cuda._wrapper")
        wrapper_module.rasterize_to_indices_in_range_2dgs = fake_indices
        meta = {
            "means2d": torch.zeros((2, 3, 2)),
            "ray_transforms": torch.zeros((2, 3, 3, 3)),
            "opacities": torch.ones((2, 3)),
            "isect_offsets": torch.zeros((2, 1, 1), dtype=torch.int32),
            "flatten_ids": torch.zeros(0, dtype=torch.int32),
            "width": 2,
            "height": 2,
            "tile_size": 16,
            "camera_ids": None,
            "gaussian_ids": None,
        }
        with patch.dict(
            sys.modules,
            {
                "gsplat": gsplat_module,
                "gsplat.cuda": cuda_module,
                "gsplat.cuda._wrapper": wrapper_module,
            },
        ):
            sequence = collect_ordered_hits(meta)
        torch.testing.assert_close(sequence.gaussian_ids, torch.tensor([1, 0, 2]))
        torch.testing.assert_close(sequence.segment_ids, torch.tensor([0, 7, 7]))
        self.assertEqual(captured["args"][0:2], (0, 2**31 - 1))
        self.assertEqual(tuple(captured["args"][2].shape), (2, 2, 2))


if __name__ == "__main__":
    unittest.main()
