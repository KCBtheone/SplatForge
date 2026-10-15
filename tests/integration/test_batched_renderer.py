from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from splatforge.studio.batched_renderer import BatchedTorchRenderer
from splatforge.studio.config import load_config
from splatforge.studio.demo import make_demo
from splatforge.studio.methods import methods
from splatforge.studio.renderers import GaussianView


class BatchedRendererTests(unittest.TestCase):
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
                    parameters = list(pipeline.model.params.values())

                    def loss(output):
                        return (
                            output.rgb.sum()
                            + output.depth.mean()
                            + output.distortion_l1.mean()
                            + output.distortion_l2.mean()
                            + output.ray_color.mean()
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
