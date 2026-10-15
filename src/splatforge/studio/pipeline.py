"""Representation-specific composition; the trainer only consumes this protocol."""

from __future__ import annotations

import math

import torch

from ..gaussian import SurfaceOperators
from .data import DataManager
from .densification import GaussianDensifier
from .model import GaussianScene
from .multiview import multiview_losses
from .renderers import GaussianView, make_renderer


class GaussianPipeline:
    def __init__(self, config, method):
        self.config = config
        self.method = method
        self.ops = SurfaceOperators()
        self.data = DataManager(config.data, config.render.background)
        if method.requires_prior:
            absent = [
                frame.image.name
                for i, frame in enumerate(self.data.frames)
                if i in self.data.train_indices and frame.prior is None
            ]
            if absent:
                raise ValueError(
                    f"AmbiSuR requires precomputed aligned depth priors; missing: {', '.join(absent[:5])}"
                )
            if config.data.init_from_priors and config.data.prior_kind != "metric":
                raise ValueError(
                    "monocular priors cannot initialize metric points; set data.init_from_priors=false"
                )
        points, colors = self.data.initialization(
            config.train.device, config.train.seed
        )
        self.model = GaussianScene(points, colors, config.model, len(self.data.frames))
        self.scene_scale = (
            torch.linalg.vector_norm(points.amax(0) - points.amin(0))
            .detach()
            .clamp_min(1e-5)
        )
        self.renderer = make_renderer(config.render, method, self.ops)
        origins = torch.stack(
            [self.data.frames[i].camera.origin for i in self.data.train_indices]
        ).to(points)
        self.camera_extent = (
            (1.1 * torch.linalg.vector_norm(origins - origins.mean(0), dim=-1).max())
            .clamp_min(1e-5)
            .detach()
        )
        self.densifier = GaussianDensifier(
            config.densify, self.model, extent=self.camera_extent
        )

    def parameters(self):
        return self.model.parameters()

    def parameter_groups(self):
        groups = self.model.parameter_groups(self.config.train)
        if self.config.train.spatial_lr_scale == "camera-radius":
            for group in groups:
                if group["name"] == "means":
                    group["lr"] *= float(self.camera_extent)
        return groups

    def render(self, camera, *, selected=None):
        return self.renderer.render(
            GaussianView.from_model(self.model), camera, selected=selected
        )

    def training_loss(self, sample, step):
        selection = self.method.selection(self.model, self.ops)
        rgb_only = getattr(self.renderer, "render_rgb", None)
        if (
            rgb_only is not None
            and self.method.photometric_only(step)
            and not self.config.loss.multiview
        ):
            output = rgb_only(
                GaussianView.from_model(self.model),
                sample["camera"],
                selected=selection,
            )
        elif hasattr(self.renderer, "render_training"):
            output = self.renderer.render_training(
                GaussianView.from_model(self.model),
                sample["camera"],
                selected=selection,
                geometric=not self.method.photometric_only(step),
            )
        else:
            output = self.render(sample["camera"], selected=selection)
        terms = self.method.losses(self, sample, output, step)
        terms.update(self.method.regularization(self, sample, output, step))
        terms.update(multiview_losses(self, sample, output, step))
        if self.config.model.appearance:
            terms["appearance_regularization"] = (
                1e-4 * self.model.appearance.square().mean()
            )
        return sum(terms.values()), terms, output

    def after_backward(self, output):
        self.densifier.observe(output, self.model)

    def after_step(self, step, optimizer):
        self.densifier.after_step(step, self.model, optimizer)

    def statistics(self):
        return {"gaussians": self.model.count}

    def update_learning_rates(self, step, optimizer):
        settings = self.config.train
        interval = self.config.model.sh_degree_interval
        degree = (
            self.config.model.sh_degree
            if interval <= 0
            else min(self.config.model.sh_degree, step // interval)
        )
        self.model.active_sh_degree.fill_(degree)
        fraction = min(step / 30000, 1)
        for group in optimizer.param_groups:
            if group["name"] == "means":
                group["lr"] = math.exp(
                    math.log(settings.means_lr) * (1 - fraction)
                    + math.log(settings.means_lr_final) * fraction
                )
                if settings.spatial_lr_scale == "camera-radius":
                    group["lr"] *= float(self.camera_extent)

    def save_representation(self, output):
        from pathlib import Path
        from ..mesh.io import write_gaussians

        path = Path(output) / "gaussians.ply"
        write_gaussians(path, self.model)
        return {"kind": "gaussians", "count": self.model.count, "path": str(path)}

    def state_dict(self):
        return {
            "model": self.model.state_dict(),
            "densifier": self.densifier.state_dict(),
            "scene_scale": self.scene_scale,
            "camera_extent": self.camera_extent,
        }

    def load_state_dict(self, state):
        model_state = state["model"]
        for name in self.model.params:
            self.model.params[name] = torch.nn.Parameter(
                model_state["params." + name].to(self.model.device)
            )
        self.model.load_state_dict(model_state)
        self.densifier.load_state_dict(state["densifier"])
        self.scene_scale = state["scene_scale"].to(self.model.device)
        if "camera_extent" in state:
            self.camera_extent = state["camera_extent"].to(self.model.device)
            self.densifier.extent = self.camera_extent
