"""Gaussian lifecycle callback preserving Adam state across representation edits."""

from __future__ import annotations

import torch
from torch import nn

from ..providers.torch.gaussian.reference import rotation_matrix


class GaussianDensifier:
    def __init__(self, config, model, *, extent=1.0):
        self.config = config
        self.extent = extent
        self.gradient = torch.zeros(model.count, device=model.device)
        self.observations = torch.zeros_like(self.gradient)

    def observe(self, output, model):
        projected = output.projected_means
        if projected is None:
            return
        gradient = (
            getattr(projected, "absgrad", None)
            if self.config.absolute_gradients
            else projected.grad
        )
        if gradient is None:
            return
        gradient = gradient.detach().reshape(-1, 2)
        if len(gradient) != model.count:
            raise ValueError(
                "renderer densification statistics do not match Gaussian count"
            )
        visible = output.projected_radii.detach().reshape(-1) > 0
        if self.config.gradient_space == "ndc":
            # All studio projected centers use pixel coordinates. Thresholds
            # from 3DGS/gsplat instead refer to normalized screen coordinates.
            gradient = gradient * gradient.new_tensor(
                [output.rgb.shape[1] / 2, output.rgb.shape[0] / 2]
            )
        norm = torch.linalg.vector_norm(gradient, dim=-1)
        self.gradient += torch.where(visible, norm, 0)
        self.observations += visible.to(norm.dtype)

    def _replace(
        self,
        model,
        optimizer,
        indices,
        children,
        *,
        child_means=None,
        child_log_scales=None,
    ):
        # Every child index refers to the PRE-EDIT representation. ParameterDict
        # visits log_scales before means; looking up model.scales inside that
        # loop would therefore address the already-pruned/reordered/new array.
        child_scales = model.scales.detach()[children].clone()
        for name, old in list(model.params.items()):
            data = old.detach()[indices].clone()
            if len(children):
                base = old.detach()[children].clone()
                if name == "means":
                    if child_means is None:
                        base += torch.randn_like(base) * child_scales * 0.5
                    else:
                        base = child_means
                elif name == "log_scales":
                    base = base - 0.5 if child_log_scales is None else child_log_scales
                data = torch.cat((data, base), 0)
            replacement = nn.Parameter(data)
            state = optimizer.state.pop(old, {})
            transformed = {}
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and value.shape == old.shape:
                    selected = value[indices].clone()
                    transformed[key] = (
                        torch.cat((selected, torch.zeros_like(value[children])), 0)
                        if len(children)
                        else selected
                    )
                else:
                    transformed[key] = value
            for group in optimizer.param_groups:
                group["params"] = [
                    replacement if parameter is old else parameter
                    for parameter in group["params"]
                ]
            model.params[name] = replacement
            if transformed:
                optimizer.state[replacement] = transformed
        self.gradient = torch.zeros(model.count, device=model.device)
        self.observations = torch.zeros_like(self.gradient)

    @torch.no_grad()
    def after_step(self, step, model, optimizer):
        config = self.config
        if not config.enabled or step >= config.stop:
            return
        if (
            config.reset_opacity_every > 0
            and step > 0
            and step % config.reset_opacity_every == 0
        ):
            ceiling = model.params["opacity_logits"].new_tensor(-2.197224577)
            model.params["opacity_logits"].clamp_(max=ceiling)
            state = optimizer.state.get(model.params["opacity_logits"], {})
            for key in ("exp_avg", "exp_avg_sq"):
                if key in state:
                    state[key].zero_()
        if step < config.start or step % config.interval:
            return
        live = model.opacities >= config.prune_opacity
        largest_scale = model.scales.amax(-1).detach()
        if step > config.reset_opacity_every > 0:
            live &= largest_scale <= config.prune_scale_fraction * self.extent
        keep = live.nonzero().squeeze(-1)
        if not len(keep):
            keep = model.opacities.argmax().reshape(1)
        average = self.gradient / self.observations.clamp_min(1)
        grow = keep[
            (self.observations[keep] > 0) & (average[keep] >= config.gradient_threshold)
        ]
        slots = max(0, config.max_gaussians - len(keep))
        if len(grow) > slots:
            grow = grow[average[grow].argsort(descending=True)[:slots]]
        clone = grow[largest_scale[grow] <= config.grow_scale_fraction * self.extent]
        split = grow[largest_scale[grow] > config.grow_scale_fraction * self.extent]
        # A small Gaussian is cloned without a random walk. A large Gaussian
        # is replaced by two rotated, smaller children; its parent is removed.
        parents = split.repeat_interleave(2)
        offsets = (
            torch.randn((len(parents), 3), device=model.device)
            * model.scales.detach()[parents]
        )
        offsets = (
            rotation_matrix(model.quats.detach()[parents]) @ offsets[..., None]
        ).squeeze(-1)
        children = torch.cat((clone, parents))
        child_means = torch.cat(
            (model.means.detach()[clone], model.means.detach()[parents] + offsets)
        )
        child_log_scales = torch.cat(
            (
                model.params["log_scales"].detach()[clone],
                (model.scales.detach()[parents] / 1.6).log(),
            )
        )
        retained = live.clone()
        retained[split] = False
        # Keep the explicit minimum survivor if opacity pruning rejected all.
        retained[keep] = True
        retained[split] = False
        self._replace(
            model,
            optimizer,
            retained.nonzero().squeeze(-1),
            children,
            child_means=child_means,
            child_log_scales=child_log_scales,
        )

    def state_dict(self):
        return {"gradient": self.gradient, "observations": self.observations}

    def load_state_dict(self, state):
        self.gradient = state["gradient"].to(self.gradient.device)
        self.observations = state["observations"].to(self.observations.device)
