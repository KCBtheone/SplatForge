"""Representation-specific composition; the trainer only consumes this protocol."""

from __future__ import annotations

import math

import torch

from ..gaussian import SurfaceOperators
from .data import DataManager
from .densification import GaussianDensifier
from .model import GaussianScene
from .multiview import multiview_losses
from .renderers import (
    GaussianView,
    RenderOutput,
    bind_configured_ordered_adjoint,
    make_renderer,
)


class GaussianPipeline:
    densifier_type = GaussianDensifier

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
        self.renderer = bind_configured_ordered_adjoint(
            make_renderer(config.render, method, self.ops)
        )
        origins = torch.stack(
            [self.data.frames[i].camera.origin for i in self.data.train_indices]
        ).to(points)
        self.camera_extent = (
            (1.1 * torch.linalg.vector_norm(origins - origins.mean(0), dim=-1).max())
            .clamp_min(1e-5)
            .detach()
        )
        self.densifier = self.densifier_type(
            config.densify, self.model, extent=self.camera_extent,
            opacity_reset_max=getattr(
                config.model, "opacity_reset_max", 0.01 if method.name == "radegs" else 0.1
            ),
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
        return self._render(self.renderer.render, camera, selected=selected)

    def _render(self, renderer, camera, **kwargs):
        view = GaussianView.from_model(self.model)
        output = renderer(view, camera, **kwargs)
        if isinstance(output, RenderOutput):
            output.bind_inputs(camera, view.means)
        return output

    def training_loss(self, sample, step):
        if self.method.name == "radegs" and not getattr(
            self.renderer, "provides_rade_normal_numerator", False
        ):
            raise RuntimeError(
                "RaDe-GS training requires a renderer that provides raw normal_numerator"
            )
        selection = self.method.selection(self.model, self.ops)
        rgb_only = getattr(self.renderer, "render_rgb", None)
        camera = sample["camera"]
        if (
            rgb_only is not None
            and self.method.photometric_only(step)
            and not self.config.loss.multiview
        ):
            output = self._render(rgb_only, camera, selected=selection)
        elif hasattr(self.renderer, "render_training"):
            output = self._render(
                self.renderer.render_training,
                camera,
                selected=selection,
                geometric=not self.method.photometric_only(step),
            )
        else:
            output = self.render(camera, selected=selection)
        request = getattr(self.method, "render_request", None)
        if request is not None:
            request(step).validate(output, camera)
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
        filter_state = (
            getattr(self.renderer, "filter_state", None)
            if self.method.name == "radegs" else None
        )
        write_gaussians(
            path, self.model,
            filter_sigma=filter_state.sigma if filter_state is not None else None,
        )
        return {
            "kind": "gaussians", "count": self.model.count, "path": str(path),
            "filter3d_baked": filter_state is not None,
        }

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
        if (
            self.densifier.gradient.shape != (self.model.count,)
            or self.densifier.observations.shape != (self.model.count,)
        ):
            raise ValueError("checkpoint densification statistics do not match Gaussian layout")
        self.scene_scale = state["scene_scale"].to(self.model.device)
        if "camera_extent" in state:
            self.camera_extent = state["camera_extent"].to(self.model.device)
            self.densifier.extent = self.camera_extent
