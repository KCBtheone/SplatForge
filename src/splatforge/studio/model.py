"""Gaussian representation with named parameter groups and appearance compensation."""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn

from .config import ModelConfig


class GaussianScene(nn.Module):
    def __init__(self, points, colors, config: ModelConfig, n_views: int):
        super().__init__()
        self.config = config
        self.register_buffer("active_sh_degree", torch.tensor(0, device=points.device))
        if len(points) > 1:
            distance, _ = cKDTree(points.detach().cpu().numpy()).query(
                points.detach().cpu().numpy(), k=min(4, len(points))
            )
            distance = np.asarray(distance)[:, 1:]
            scale = np.maximum(np.sqrt(np.mean(distance**2, axis=1)), 1e-5)
        else:
            scale = np.ones(len(points)) * 0.01
        scales = (
            torch.tensor(scale, device=points.device, dtype=points.dtype)
            .unsqueeze(-1)
            .expand(-1, 3)
            * config.initial_scale
        )
        quats = torch.zeros((len(points), 4), device=points.device, dtype=points.dtype)
        quats[:, 0] = 1
        sh = torch.zeros(
            (len(points), (config.sh_degree + 1) ** 2, 3),
            device=points.device,
            dtype=points.dtype,
        )
        sh[:, 0] = (colors - 0.5) / math.sqrt(1 / (4 * math.pi))
        self.params = nn.ParameterDict(
            {
                "means": nn.Parameter(points.clone()),
                "log_scales": nn.Parameter(scales.log()),
                "quats": nn.Parameter(quats),
                "opacity_logits": nn.Parameter(
                    torch.full(
                        (len(points),),
                        math.log(config.opacity / (1 - config.opacity)),
                        device=points.device,
                        dtype=points.dtype,
                    )
                ),
                "sh": nn.Parameter(sh),
            }
        )
        self.appearance = nn.Parameter(
            torch.zeros((n_views, 6), device=points.device, dtype=points.dtype),
            requires_grad=config.appearance,
        )

    @property
    def means(self):
        return self.params["means"]

    @property
    def quats(self):
        return self.params["quats"]

    @property
    def scales(self):
        return self.params["log_scales"].exp()

    @property
    def opacities(self):
        return self.params["opacity_logits"].sigmoid()

    @property
    def sh(self):
        return self.params["sh"]

    @property
    def count(self):
        return len(self.means)

    @property
    def device(self):
        return self.means.device

    def parameter_groups(self, train_config):
        settings = {
            "means": train_config.means_lr,
            "log_scales": train_config.scales_lr,
            "quats": train_config.quats_lr,
            "opacity_logits": train_config.opacity_lr,
            "sh": train_config.features_lr,
        }
        groups = [
            {"params": [self.params[name]], "lr": lr, "name": name}
            for name, lr in settings.items()
        ]
        if self.appearance.requires_grad:
            groups.append(
                {
                    "params": [self.appearance],
                    "lr": train_config.appearance_lr,
                    "name": "appearance",
                }
            )
        return groups

    def compensate(self, image, view_index):
        if not self.config.appearance:
            return image
        affine = self.appearance[view_index]
        return image * affine[:3].exp() + affine[3:]
