from __future__ import annotations

import unittest

import torch

from splatforge.studio.config import DensifyConfig, ModelConfig, TrainConfig
from splatforge.studio.densification import GaussianDensifier
from splatforge.studio.model import GaussianScene
from splatforge.studio.renderers import RenderOutput


class DensificationTests(unittest.TestCase):
    def fixture(self, scales):
        points = torch.arange(len(scales) * 3, dtype=torch.float32).reshape(-1, 3)
        model = GaussianScene(points, torch.ones_like(points) * 0.5, ModelConfig(), 1)
        with torch.no_grad():
            model.params["log_scales"].copy_(torch.tensor(scales).log())
        config = DensifyConfig(start=1, interval=1, reset_opacity_every=0)
        config.gradient_threshold = 0.0
        optimizer = torch.optim.Adam(model.parameter_groups(TrainConfig()))
        densifier = GaussianDensifier(config, model, extent=10.0)
        densifier.observations.fill_(1)
        return model, optimizer, densifier

    def test_small_gaussian_is_cloned_without_scale_shrink_or_random_offset(self):
        model, optimizer, densifier = self.fixture([[0.01, 0.02, 0.03]])
        old_means, old_scales = (
            model.means.detach().clone(),
            model.scales.detach().clone(),
        )
        densifier.after_step(1, model, optimizer)
        self.assertEqual(model.count, 2)
        torch.testing.assert_close(model.means, old_means.repeat(2, 1))
        torch.testing.assert_close(model.scales, old_scales.repeat(2, 1))

    def test_large_parent_is_replaced_by_two_rotated_smaller_children(self):
        model, optimizer, densifier = self.fixture([[0.2, 0.3, 0.4]])
        with torch.no_grad():
            model.quats.copy_(torch.tensor([[0.70710678, 0, 0, 0.70710678]]))
        original = model.means.detach().clone()
        scales = model.scales.detach().clone()
        torch.manual_seed(18)
        random = torch.randn(2, 3) * scales
        expected = original + torch.stack(
            (-random[:, 1], random[:, 0], random[:, 2]), -1
        )
        torch.manual_seed(18)
        densifier.after_step(1, model, optimizer)
        self.assertEqual(model.count, 2)
        torch.testing.assert_close(model.means, expected)
        torch.testing.assert_close(model.scales, scales.repeat(2, 1) / 1.6)

    def test_invisible_gaussian_does_not_grow_even_at_zero_threshold(self):
        model, optimizer, densifier = self.fixture([[0.2, 0.3, 0.4]])
        densifier.observations.zero_()
        densifier.after_step(1, model, optimizer)
        self.assertEqual(model.count, 1)

    def test_normalized_screen_gradient_uses_actual_image_dimensions(self):
        model, _, densifier = self.fixture([[0.2, 0.3, 0.4]])
        densifier.observations.zero_()
        xy = torch.zeros(1, 2, requires_grad=True)
        xy.grad = torch.tensor([[0.001, 0.002]])
        output = RenderOutput(
            torch.zeros(20, 40, 3),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            xy,
            torch.ones(1),
        )
        densifier.observe(output, model)
        torch.testing.assert_close(
            densifier.gradient, torch.tensor([(0.02**2 + 0.02**2) ** 0.5])
        )

    def test_children_use_pre_edit_scales_when_pruning_and_reordering(self):
        points = torch.arange(15, dtype=torch.float32).reshape(5, 3)
        model = GaussianScene(points, torch.ones_like(points) * 0.5, ModelConfig(), 1)
        with torch.no_grad():
            model.params["log_scales"].copy_(
                torch.arange(1, 6, dtype=torch.float32)[:, None].expand(-1, 3).log()
            )
        optimizer = torch.optim.Adam(model.parameter_groups(TrainConfig()))
        sum(p.square().sum() for p in model.params.values()).backward()
        optimizer.step()
        old_means = model.means.detach().clone()
        old_scales = model.scales.detach().clone()
        old_moments = optimizer.state[model.means]["exp_avg"].clone()
        kept = torch.tensor([4, 1])
        children = torch.tensor([4])  # old index 4 exceeds the new array length 3
        torch.manual_seed(51)
        expected = old_means[children] + torch.randn(1, 3) * old_scales[children] * 0.5
        torch.manual_seed(51)
        densifier = GaussianDensifier(DensifyConfig(), model)
        densifier._replace(model, optimizer, kept, children)
        self.assertEqual(model.count, 3)
        torch.testing.assert_close(model.means[:2], old_means[kept])
        torch.testing.assert_close(model.means[2:], expected)
        torch.testing.assert_close(
            model.scales[2:], old_scales[children] * torch.exp(torch.tensor(-0.5))
        )
        torch.testing.assert_close(
            optimizer.state[model.means]["exp_avg"][:2], old_moments[kept]
        )
        torch.testing.assert_close(
            optimizer.state[model.means]["exp_avg"][2:], torch.zeros(1, 3)
        )
        self.assertEqual(densifier.gradient.shape, (3,))
