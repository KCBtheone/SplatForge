"""Small-memory fused-backend tests runnable under GPU contention.

All cases use M<=4096 hits (<1 MB) so they pass on CPU and on a shared
RTX 4080 SUPER with only ~2 GiB free. They verify explicit backend
accounting, Torch-fallback bitwise equality, and structured errors --
never silent substitution. Source: SPEC.md equations only.
"""

from __future__ import annotations

import unittest

import torch

from unbiased_gsplat._cuda import (
    cuda_available,
    get_build_info,
    ray_splat_depth_and_weight_fused,
)
from unbiased_gsplat.ray_splat_depth import (
    ray_splat_depth_and_weight,
    ray_splat_depth_and_weight_torch,
)


def _tiny_batch(device: str = "cpu"):
    # 2 pixels, 4 hits: one center hit, one offset hit, one degenerate, one far.
    ray_transforms = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 5.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.0, 5.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 6.0]],
        ],
        dtype=torch.float32,
        device=device,
    )
    means2d = torch.tensor(
        [[0.5, 0.5], [2.0, 0.5], [0.5, 0.5], [10.0, 10.0]],
        dtype=torch.float32,
        device=device,
    )
    pixel_ids = torch.tensor([0, 0, 1, 1], dtype=torch.int64, device=device)
    return ray_transforms, means2d, pixel_ids


class FusedBackendTests(unittest.TestCase):
    def test_auto_reports_torch_fallback_when_unbuilt(self):
        rt, m2, pid = _tiny_batch()
        _, backend = ray_splat_depth_and_weight_fused(rt, m2, pid, 2, backend="auto")
        if cuda_available() and rt.is_cuda:
            self.assertEqual(backend, "cuda-fused")
        else:
            self.assertEqual(backend, "torch-fallback")

    def test_cuda_matches_torch_forward_and_grad_when_built(self):
        if not cuda_available():
            self.skipTest("fused CUDA extension is not built")
        rt, m2, pid = _tiny_batch("cuda")
        rt = rt.detach().requires_grad_(True)
        m2 = m2.detach().requires_grad_(True)
        ref = ray_splat_depth_and_weight_torch(rt, m2, pid, 2)
        out, backend = ray_splat_depth_and_weight_fused(rt, m2, pid, 2, backend="cuda")
        self.assertEqual(backend, "cuda-fused")
        self.assertEqual(out.valid_mask.tolist(), ref.valid_mask.tolist())
        self.assertEqual(out.ray_plane_mask.tolist(), ref.ray_plane_mask.tolist())
        torch.testing.assert_close(out.gaussian_weight, ref.gaussian_weight, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(out.depth, ref.depth, atol=1e-5, rtol=1e-5)
        (out.gaussian_weight.sum() + out.depth.sum()).backward()
        g_rt_cuda, g_m2_cuda = rt.grad.clone(), m2.grad.clone()
        rt.grad = None
        m2.grad = None
        (ref.gaussian_weight.sum() + ref.depth.sum()).backward()
        torch.testing.assert_close(g_rt_cuda, rt.grad, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(g_m2_cuda, m2.grad, atol=1e-4, rtol=1e-4)

    def test_torch_backend_bitwise_matches_normative(self):
        rt, m2, pid = _tiny_batch()
        ref = ray_splat_depth_and_weight(rt, m2, pid, 2)
        out, backend = ray_splat_depth_and_weight_fused(rt, m2, pid, 2, backend="torch")
        self.assertEqual(backend, "torch-fallback")
        self.assertTrue(torch.equal(out.valid_mask, ref.valid_mask))
        self.assertTrue(torch.equal(out.gaussian_weight, ref.gaussian_weight))
        self.assertTrue(torch.equal(out.depth, ref.depth))

    def test_cuda_backend_raises_structured_when_unbuilt(self):
        rt, m2, pid = _tiny_batch()
        if cuda_available():
            self.skipTest("built CUDA present; stub raise-test not applicable")
        with self.assertRaises(RuntimeError) as ctx:
            ray_splat_depth_and_weight_fused(rt, m2, pid, 2, backend="cuda")
        self.assertIn("not built", str(ctx.exception))

    def test_corrected_depth_and_convergence_cuda_run(self):
        if not cuda_available():
            self.skipTest("fused CUDA extension is not built")
        device = "cuda"
        depths = torch.tensor([1.0, 1.2, 9.0, 4.0], device=device, requires_grad=True)
        weights = torch.tensor([0.7, 0.7, 0.7, 0.7], device=device)
        opacities = torch.ones(4, device=device)
        seg = torch.tensor([0, 0, 1, 1], device=device)
        valid = torch.ones(4, dtype=torch.bool, device=device)
        from unbiased_gsplat.corrected_depth import corrected_depth
        from unbiased_gsplat.convergence import depth_convergence

        cuda_cd = corrected_depth(
            depths, opacities, weights, seg, valid, num_segments=3, threshold=0.5
        )
        self.assertTrue(bool(cuda_cd.valid_mask[0].item()))
        cuda_cd.depth.sum().backward()
        self.assertIsNotNone(depths.grad)
        conv = depth_convergence(
            depths.detach().requires_grad_(True),
            weights,
            seg,
            valid,
            scene_radius=10.0,
            num_segments=3,
            reduction="sum",
        )
        conv.loss.backward()

    def test_zncc_and_depth_to_points_cuda_grad(self):
        if not cuda_available():
            self.skipTest("fused CUDA extension is not built")
        a = torch.randn(8, 49, device="cuda", dtype=torch.float32, requires_grad=True)
        b = a.detach() * 0.5 + 0.1
        b.requires_grad_(True)
        from unbiased_gsplat.multiview import normalized_cross_correlation
        from unbiased_gsplat.depth_normal import depth_to_points

        corr, valid = normalized_cross_correlation(a, b)
        (corr * valid.to(corr.dtype)).sum().backward()
        self.assertIsNotNone(a.grad)
        depth = torch.full((1, 5, 5), 2.0, device="cuda", dtype=torch.float32, requires_grad=True)
        valid_d = torch.ones((1, 5, 5), dtype=torch.bool, device="cuda")
        c2w = torch.eye(4, device="cuda", dtype=torch.float32)[None]
        K = torch.tensor(
            [[[100.0, 0.0, 2.5], [0.0, 100.0, 2.5], [0.0, 0.0, 1.0]]],
            device="cuda",
            dtype=torch.float32,
        )
        points, _ = depth_to_points(depth, valid_d, c2w, K)
        points.sum().backward()
        self.assertIsNotNone(depth.grad)

    def test_small_memory_footprint(self):
        rt, m2, pid = _tiny_batch()
        out, _ = ray_splat_depth_and_weight_fused(rt, m2, pid, 2, backend="torch")
        total = out.depth.numel() * 4 + out.gaussian_weight.numel() * 4
        self.assertLessEqual(total, 1 << 20)


if __name__ == "__main__":
    unittest.main()
