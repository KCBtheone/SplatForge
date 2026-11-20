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
        if config.paper_sh_lr_split:
            # Ordinary Adam needs separate parameters for distinct learning
            # rates. Scaling a gradient would also scale Adam's moments and
            # would not implement the original DC/rest learning-rate ratio.
            original_sh = self.params.pop("sh")
            self.params["sh_dc"] = nn.Parameter(original_sh.detach()[:, :1].contiguous())
            self.params["sh_rest"] = nn.Parameter(original_sh.detach()[:, 1:].contiguous())
        self.appearance = nn.Parameter(
            torch.zeros((n_views, 6), device=points.device, dtype=points.dtype),
            requires_grad=config.appearance,
        )
        if config.paper_decoupled_appearance:
            from ..integrations.radegs_appearance import DecoupledAppearance
            # The CNN is initialized on CPU. Preserve both the original CPU
            # Gaussian RNG stream and the untouched CUDA densification stream.
            with torch.random.fork_rng(devices=[]):
                self.decoupled_appearance = DecoupledAppearance(n_views)
            self.decoupled_appearance.to(device=points.device,dtype=points.dtype)

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
        if self.config.paper_sh_lr_split:
            return torch.cat((self.params["sh_dc"], self.params["sh_rest"]), dim=1)
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
        if self.config.paper_sh_lr_split:
            settings.pop("sh")
            settings.update(sh_dc=train_config.features_lr, sh_rest=train_config.features_lr / 20)
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
        if self.config.paper_decoupled_appearance:
            module=self.decoupled_appearance
            groups.extend([
                {"params":list(module.embedding.parameters()),"lr":.001,"name":"appearance_embedding"},
                {"params":list(module.upsample.parameters())+list(module.output.parameters()),
                 "lr":.001,"name":"appearance_network"},
            ])
        return groups

    def compensate(self, image, view_index):
        if not self.config.appearance:
            return image
        affine = self.appearance[view_index]
        return image * affine[:3].exp() + affine[3:]
