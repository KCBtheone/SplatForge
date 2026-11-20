"""Native baseline selection and its promotion boundary are explicit."""

from unittest.mock import Mock

from splatforge.integrations.gsplat_radegs_moments import GsplatRadeMomentRenderer
from splatforge.studio.config import RenderConfig
from splatforge.studio.methods.radegs import RaDeGS


def test_native_geometry_failure_does_not_promote_experimental_default():
    config = RaDeGS.default_config()
    assert config.render.backend == "gsplat-radegs"
    assert config.loss.geometric_start == 15000
    assert config.loss.depth_distortion == 100.0
    assert config.train.steps == 30000


def test_native_geometry_returns_upstream_outputs_without_stable_operator():
    from types import SimpleNamespace
    import torch

    renderer = GsplatRadeMomentRenderer.__new__(GsplatRadeMomentRenderer)
    renderer.config = RenderConfig()
    renderer.stable_geometry = False
    renderer.cuda_geometry = False
    renderer.ops = Mock(
        side_effect=AssertionError("native must not call self-developed geometry")
    )
    outputs = tuple(object() for _ in range(8))
    renderer.project = Mock(return_value=outputs)
    scene = SimpleNamespace(
        means=torch.zeros(1, 3),
        quats=torch.zeros(1, 4),
        scales=torch.ones(1, 3),
        opacities=torch.ones(1),
    )
    camera = SimpleNamespace(
        world_to_camera=torch.eye(4), K=torch.eye(3), width=8, height=8
    )
    assert renderer.geometry(scene, camera) is outputs
    assert renderer.project.call_args.kwargs["radegs"] is True
    renderer.ops.assert_not_called()
