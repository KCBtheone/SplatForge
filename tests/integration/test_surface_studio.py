from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import torch

from splatforge.mesh.io import read_vertices, write_mesh
from splatforge.mesh.tetrahedra import marching_tetrahedra, triangulate
from splatforge.studio.config import load_config
from splatforge.studio.demo import make_demo
from splatforge.studio.methods import methods
from splatforge.studio.renderers import GaussianView
from splatforge.studio.trainer import Trainer, load_run
from splatforge.studio.viewer import make_server


class StudioIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = make_demo(
            self.root / "data", width=16, height=16, points=16, views=3
        )

    def config(self, method, output="run", steps=2):
        return load_config(
            method=method,
            overrides=[
                f"data.source={json.dumps(str(self.data))}",
                f"train.output={json.dumps(str(self.root / output))}",
                "train.device=cpu",
                "render.backend=torch",
                "data.downscale=1",
                "data.init_points=16",
                f"train.steps={steps}",
                "train.log_every=100",
                "train.eval_every=0",
                "train.checkpoint_every=0",
                "train.export_on_finish=false",
                "densify.enabled=false",
                "model.opacity=0.6",
                "loss.geometric_start=0",
                "loss.ambiguity_start=0",
            ],
        )

    def test_three_methods_train_save_load_and_render(self):
        for name in methods.names():
            with self.subTest(method=name):
                config = self.config(name, output=name)
                pipeline = Trainer(config).run()
                loaded, step = load_run(config.train.output, device="cpu")
                self.assertEqual(step, 2)
                sample = pipeline.data.sample(0)
                with torch.no_grad():
                    a = pipeline.render(sample["camera"])
                    b = loaded.render(sample["camera"])
                torch.testing.assert_close(a.rgb, b.rgb)
                self.assertTrue(torch.isfinite(a.depth).all())
                self.assertTrue(
                    (Path(config.train.output) / "evaluation_test.json").is_file()
                )

    def test_resume_restores_adam_rng_and_sampler_exactly(self):
        complete = Trainer(self.config("gof", output="complete", steps=4)).run()
        Trainer(self.config("gof", output="split", steps=2)).run()
        resumed = Trainer(
            self.config("gof", output="split", steps=4), resume="latest"
        ).run()
        for name, value in complete.model.state_dict().items():
            torch.testing.assert_close(
                value, resumed.model.state_dict()[name], rtol=0, atol=0
            )

    def test_isolated_ambiguity_geometry_has_exact_gradient_boundary(self):
        pipeline = methods.get("ambisur").make_pipeline(self.config("ambisur"))
        selected = torch.zeros(pipeline.model.count, dtype=torch.bool)
        selected[0] = True
        isolated = GaussianView.from_model(pipeline.model).isolated_geometry(selected)
        loss = (
            isolated.means.sum()
            + isolated.quats.sum()
            + isolated.scales.sum()
            + isolated.opacities.sum()
            + isolated.sh.sum()
        )
        loss.backward()
        torch.testing.assert_close(
            pipeline.model.means.grad[1:],
            torch.zeros_like(pipeline.model.means.grad[1:]),
        )
        self.assertIsNone(pipeline.model.params["log_scales"].grad)
        self.assertIsNone(pipeline.model.params["opacity_logits"].grad)
        self.assertIsNone(pipeline.model.sh.grad)

    def test_mesh_bisection_and_ply_roundtrip_on_analytic_plane(self):
        points = np.asarray(
            [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)]
        )

        def field(p):
            return p[:, 2] - 0.25

        vertices, faces = marching_tetrahedra(
            points,
            triangulate(points),
            field(points),
            field,
            binary_steps=12,
            inside_greater=False,
        )
        self.assertGreater(len(faces), 0)
        self.assertLess(float(np.abs(vertices[:, 2] - 0.25).max()), 0.001)
        path = self.root / "mesh.ply"
        write_mesh(path, vertices, faces)
        np.testing.assert_allclose(read_vertices(path), vertices, atol=1e-7)

    def test_densification_keeps_optimizer_shapes_and_checkpoint_consistent(self):
        config = self.config("gof", steps=3)
        config.densify.enabled = True
        config.densify.start = 1
        config.densify.interval = 1
        config.densify.gradient_threshold = 0.0
        config.densify.max_gaussians = 32
        pipeline = Trainer(config).run()
        self.assertEqual(pipeline.model.count, 32)
        loaded, _ = load_run(config.train.output, device="cpu")
        self.assertEqual(loaded.model.count, 32)

    def test_missing_priors_and_unknown_config_are_errors(self):
        with self.assertRaises(ValueError):
            load_config(method="gof", overrides=["unknown.value=3"])
        for path in (self.data / "priors").glob("*.npz"):
            path.unlink()
        with self.assertRaisesRegex(ValueError, "requires precomputed"):
            methods.get("ambisur").make_pipeline(self.config("ambisur"))

    def test_viewer_state_cloud_render_and_invalid_view(self):
        config = self.config("gof")
        Trainer(config).run()
        server = make_server(config.train.output, port=0, device="cpu")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        url = f"http://127.0.0.1:{server.server_address[1]}"
        with urllib.request.urlopen(url + "/state") as response:
            self.assertEqual(json.load(response)["stage"], "TRAINING_COMPLETE")
        with urllib.request.urlopen(url + "/cloud") as response:
            self.assertEqual(json.load(response)["count"], 16)
        write_mesh(
            Path(config.train.output) / "mesh.ply",
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            np.array([[0, 1, 2]]),
        )
        with urllib.request.urlopen(url + "/mesh") as response:
            self.assertEqual(json.load(response)["count"], 1)
        for resource in ("viewer.css", "viewer.js"):
            with urllib.request.urlopen(url + "/" + resource) as response:
                self.assertEqual(response.status, 200)
        with urllib.request.urlopen(url + "/render?view=0&mode=normal") as response:
            self.assertEqual(response.read()[:8], b"\x89PNG\r\n\x1a\n")
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(url + "/render?view=9999")
        self.assertEqual(error.exception.code, 400)
        error.exception.close()
        camera = Trainer(self.config("gof", output="camera-pipeline")).config
        from splatforge.studio.data import DataManager

        view = DataManager(camera.data).sample(0)["camera"]
        payload = {
            "K": view.K.tolist(),
            "world_to_camera": view.world_to_camera.tolist(),
            "width": 16,
            "height": 16,
            "mode": "rgb",
        }
        request = urllib.request.Request(
            url + "/render-camera",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request) as response:
            self.assertEqual(response.read()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
