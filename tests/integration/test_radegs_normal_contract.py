from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from splatforge.integrations.gsplat_radegs_moments import GsplatRadeMomentRenderer
from splatforge.integrations.radegs_filter3d import FilteredRenderer
from splatforge.studio.methods.radegs import RaDeGS
from splatforge.studio.pipeline import GaussianPipeline


def test_actual_backend_rejected_before_render_or_update():
    pipeline = GaussianPipeline.__new__(GaussianPipeline)
    pipeline.method = RaDeGS(RaDeGS.default_config())
    pipeline.renderer = SimpleNamespace(render=Mock())
    with pytest.raises(RuntimeError, match="renderer.*raw normal_numerator"):
        pipeline.training_loss({}, 0)
    pipeline.renderer.render.assert_not_called()


def test_supported_wrapped_backend_allows_rgb_only_photo():
    pipeline = GaussianPipeline.__new__(GaussianPipeline)
    config = RaDeGS.default_config()
    pipeline.config = config
    pipeline.method = RaDeGS(config)
    pipeline.method.losses = Mock(return_value={"photo": torch.tensor(1.0)})
    pipeline.ops = object()
    pipeline.model = SimpleNamespace(appearance=None)
    output = SimpleNamespace(rgb=torch.tensor(1.0))
    backend = GsplatRadeMomentRenderer.__new__(GsplatRadeMomentRenderer)
    backend.render_rgb = Mock(return_value=output)
    state = SimpleNamespace(counts={"filter3d_render": 0}, apply=lambda scene, track: scene)
    pipeline.renderer = FilteredRenderer(backend, state)
    with patch("splatforge.studio.pipeline.GaussianView.from_model", return_value=object()), \
         patch("splatforge.studio.pipeline.multiview_losses", return_value={}):
        loss, _, actual = pipeline.training_loss({"camera": object()}, 0)
    assert loss.item() == 1.0 and actual is output
    backend.render_rgb.assert_called_once()


def test_geometry_requires_actual_numerator_and_preserves_weighted_gradient():
    config = RaDeGS.default_config()
    method = RaDeGS(config)
    normal = torch.tensor([[[0.0, 0.0, 1.0]]])
    pipeline = SimpleNamespace(ops=lambda *args: (normal, torch.ones((1, 1), dtype=torch.bool)))
    sample = {"camera": SimpleNamespace(K=torch.eye(3)), "mask": torch.ones(1, 1, 1)}
    output = SimpleNamespace(distortion_l2=torch.zeros(1), depth=torch.ones(1, 1, 1),
                             alpha=torch.tensor([[[0.8]]]), normals=normal,
                             normal_numerator=None)
    with pytest.raises(RuntimeError, match="geometry.*raw normal_numerator"):
        method.regularization(pipeline, sample, output, 15000)
    # Opposing contributions partially cancel; their unnormalized sum is 0.2.
    weights = torch.tensor([0.5, 0.3], requires_grad=True)
    directions = torch.tensor([[0., 0., 1.], [0., 0., -1.]])
    output.normal_numerator = (weights[:, None] * directions).sum(0).reshape(1, 1, 3)
    loss = method.regularization(pipeline, sample, output, 15000)["normal_consistency"]
    loss.backward()
    torch.testing.assert_close(loss, torch.tensor(3.0))
    torch.testing.assert_close(weights.grad, torch.tensor([-5.0, 5.0]))
