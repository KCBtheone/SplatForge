import torch
import torch.nn.functional as F

from ..losses import masked_mean, normal_consistency, patch_depth_loss
from ..renderers import GaussianView
from .base import SurfaceMethod


class AmbiSuR(SurfaceMethod):
    name = "ambisur"
    description = "Truncated planar GS, color-only ray consistency, SH dual-end indication and isolated local geometry priors"
    rendering_mode = "ambisur"
    requires_prior = True

    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.data.init_from_priors = True
        config.loss.depth_distortion = 0.0
        config.loss.normal = 0.05
        return config

    def selection(self, model, ops):
        with torch.no_grad():
            score = ops("gaussian.sh_ambiguity@1", model.sh)
            upper = self.config.loss.ambiguity_upper
            lower = self.config.loss.ambiguity_lower
            return (
                (score > torch.quantile(score, 1 - upper))
                if upper > 0
                else torch.zeros_like(score, dtype=torch.bool)
            ) | (
                (score < torch.quantile(score, lower))
                if lower > 0
                else torch.zeros_like(score, dtype=torch.bool)
            )

    def regularization(self, pipeline, sample, output, step):
        settings = self.config.loss
        terms = {}
        if step < settings.ray_color_stop:
            terms["ray_color"] = settings.ray_color * output.ray_color.mean()
        if step >= settings.geometric_start:
            prior = sample["prior_depth"]
            valid = (prior > 0) & torch.isfinite(prior) & (sample["mask"] > 0)
            if self.config.data.prior_kind == "metric":
                depth_loss = (
                    masked_mean((output.depth - prior).abs(), valid)
                    / pipeline.scene_scale
                )
            else:
                depth_loss = patch_depth_loss(output.depth, prior, valid)
            terms["prior_depth"] = settings.prior_depth * depth_loss
            terms["normal_consistency"] = settings.normal * normal_consistency(
                output, sample["camera"], pipeline.ops, sample["mask"]
            )
        if step >= settings.ambiguity_start:
            selected = self.selection(pipeline.model, pipeline.ops)
            isolated = GaussianView.from_model(pipeline.model).isolated_geometry(
                selected
            )
            if hasattr(pipeline.renderer, "render_training"):
                geometry = pipeline.renderer.render_training(
                    isolated, sample["camera"], selected=selected
                )
            else:
                geometry = pipeline.renderer.render(
                    isolated, sample["camera"], selected=selected
                )
            normal, valid = pipeline.ops(
                "geometry.depth_normals@1",
                geometry.depth.squeeze(-1),
                sample["camera"].K,
            )
            if "prior_normal" in sample:
                target = F.normalize(sample["prior_normal"], dim=-1)
                target_valid = torch.isfinite(target).all(-1) & (
                    target.square().sum(-1) > 0
                )
            else:
                target, target_valid = pipeline.ops(
                    "geometry.depth_normals@1",
                    sample["prior_depth"].squeeze(-1),
                    sample["camera"].K,
                )
            error = 1 - (normal * target).sum(-1)
            gate = (
                geometry.ambiguity_mask.squeeze(-1).detach()
                * valid
                * target_valid
                * (sample["mask"].squeeze(-1) > 0)
            )
            terms["ambiguity_normal"] = (
                settings.ambiguity_normal * (error * gate).mean()
            )
        return terms
