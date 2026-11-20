from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.checkpoint import checkpoint

from splatforge.studio.batched_renderer import BatchedTorchRenderer
from splatforge.studio.config import load_config
from splatforge.studio.demo import make_demo
from splatforge.studio.methods import methods
from splatforge.studio.renderers import GaussianView


class BatchedRendererTests(unittest.TestCase):
    def test_four_stream_guard_does_not_require_alternate_event_backend(self):
        from types import SimpleNamespace
        renderer=BatchedTorchRenderer(SimpleNamespace(eps2d=.3,near=.01),mode='gof')
        renderer.batch_streams=4
        self.assertFalse(renderer.cuda_events)
        scene=SimpleNamespace(means=SimpleNamespace(device=SimpleNamespace(type='cuda')))
        # CPU dispatch regression for the exact rejected premise, not CUDA
        # numerical acceptance. The original eager equation body must remain selected.
        with patch('splatforge.studio.batched_renderer.project_geometry',side_effect=RuntimeError('original eager path reached')):
            with self.assertRaisesRegex(RuntimeError,'original eager path reached'):
                renderer.render(scene,None)
        self.assertFalse(renderer.cuda_events)

    def test_four_stream_mode_is_explicit_cuda_gof_only_without_fallback(self):
        from types import SimpleNamespace
        renderer=BatchedTorchRenderer(SimpleNamespace(),mode='gof')
        self.assertEqual(renderer.batch_streams,0)
        scene=SimpleNamespace(means=torch.zeros(1,3))
        for count in (3,4):
            renderer.batch_streams=count
            renderer.cuda_events=True
            with self.assertRaisesRegex(ValueError,'no fallback'):
                renderer.render(scene,None)

    def test_cuda_event_path_keeps_outer_checkpoint(self):
        # CPU fixture checks dispatch policy only. Actual CUDA equations/VJPs
        # are checked separately on the server; there is no CPU fallback here.
        with tempfile.TemporaryDirectory() as directory:
            data = make_demo(Path(directory) / "data", width=17, points=16, views=3)
            config = load_config(
                method="gof",
                overrides=[
                    f"data.source={data}",
                    "train.device=cpu",
                    "data.downscale=1",
                ],
            )
            pipeline = methods.get("gof").make_pipeline(config)
            renderer = BatchedTorchRenderer(config.render, mode="gof", ops=pipeline.ops)
            renderer.cuda_events = True

            def fake_events(ids, valid, rays, pixels, xy, *rest):
                return xy.sum() * torch.ones(ids.shape[0], rays.shape[1], 16)

            with (
                patch.object(renderer, "_render_cuda_batch", side_effect=fake_events),
                patch(
                    "splatforge.studio.batched_renderer.checkpoint", wraps=checkpoint
                ) as checkpoint_call,
            ):
                output = renderer.render_training(
                    GaussianView.from_model(pipeline.model),
                    pipeline.data.sample(0)["camera"],
                )
                output.rgb.sum().backward()
                self.assertGreater(checkpoint_call.call_count, 0)

    def test_forward_and_parameter_gradients_match_reference_across_tiles(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            data = make_demo(
                Path(directory) / "data", width=33, height=21, points=16, views=3
            )
            for name in ("radegs", "ambisur", "gof"):
                with self.subTest(method=name):
                    config = load_config(
                        method=name,
                        overrides=[
                            f"data.source={data}",
                            "train.device=cpu",
                            "render.backend=torch",
                            "data.downscale=1",
                            "data.init_points=16",
                        ],
                    )
                    pipeline = methods.get(name).make_pipeline(config)
                    scene = GaussianView.from_model(pipeline.model)
                    camera = pipeline.data.sample(0)["camera"]
                    selected = torch.arange(pipeline.model.count) % 2 == 0
                    a = pipeline.renderer.render(scene, camera, selected=selected)
                    b = BatchedTorchRenderer(
                        config.render, mode=name, ops=pipeline.ops
                    ).render(scene, camera, selected=selected)
                    for key in (
                        "rgb",
                        "alpha",
                        "depth",
                        "normals",
                        "distortion_l1",
                        "distortion_l2",
                        "median_depth",
                        "ray_color",
                        "ambiguity_mask",
                    ):
                        torch.testing.assert_close(
                            getattr(a, key), getattr(b, key), atol=2e-5, rtol=2e-5
                        )
                    if name == "gof":
                        self.assertIsNotNone(a.normal_numerator)
                        self.assertIsNotNone(b.normal_numerator)
                        torch.testing.assert_close(a.normal_numerator, b.normal_numerator, atol=2e-5, rtol=2e-5)
                        torch.testing.assert_close(torch.nn.functional.normalize(a.normal_numerator, dim=-1), a.normals)
                    else:
                        self.assertIsNone(a.normal_numerator)
                        self.assertIsNone(b.normal_numerator)
                    parameters = list(pipeline.model.params.values())

                    def loss(output):
                        return (
                            output.rgb.sum()
                            + output.depth.mean()
                            + output.distortion_l1.mean()
                            + output.distortion_l2.mean()
                            + output.ray_color.mean()
                            + (output.normal_numerator.square().mean() if output.normal_numerator is not None else 0)
                        )

                    ga = torch.autograd.grad(loss(a), parameters, retain_graph=True)
                    gb = torch.autograd.grad(loss(b), parameters)
                    for first, second in zip(ga, gb):
                        torch.testing.assert_close(first, second, atol=5e-4, rtol=5e-4)

    def test_ray_color_has_no_geometry_gradient(self):
        with tempfile.TemporaryDirectory() as directory:
            data = make_demo(Path(directory) / "data", width=17, points=16, views=3)
            config = load_config(
                method="ambisur",
                overrides=[
                    f"data.source={data}",
                    "train.device=cpu",
                    "render.backend=torch-batched",
                    "data.downscale=1",
                    "data.init_points=16",
                ],
            )
            pipeline = methods.get("ambisur").make_pipeline(config)
            pipeline.model.active_sh_degree.fill_(1)
            with torch.no_grad():
                pipeline.model.sh[:, 1:].normal_()
            output = pipeline.render(pipeline.data.sample(0)["camera"])
            output.ray_color.sum().backward()
            self.assertGreater(float(pipeline.model.sh.grad.abs().sum()), 0)
            for parameter in (
                pipeline.model.means,
                pipeline.model.quats,
                pipeline.model.params["opacity_logits"],
            ):
                if parameter.grad is not None:
                    torch.testing.assert_close(
                        parameter.grad, torch.zeros_like(parameter.grad), rtol=0, atol=0
                    )
