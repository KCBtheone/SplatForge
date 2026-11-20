"""VCR-GauS (Chen et al., 2024), with depth, normal and distortion losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..losses import photometric_loss
from ..vcr_config import VCRStudioConfig
from ..vcr_losses import (
    cosine_confidence,
    edge_aware_weight,
    normal_curvature,
    normal_l1_cosine,
    reference_ssim_loss,
)
from .base import SurfaceMethod


class VCRGauS(SurfaceMethod):
    name = "vcrgaus"
    description = "VCR-GauS view-consistent depth-normal regularization"
    rendering_mode = "vcrgaus"
    exporter = "vcr-tsdf"

    @classmethod
    def default_config(cls):
        config = VCRStudioConfig()
        config.render.backend = "vcr-native"
        config.render.far = 100.0
        config.data.prior_kind = "mono"
        config.data.test_every = 0
        config.export.tsdf_backend = "open3d"
        config.export.truncation = 0.02
        config.export.depth_max = 3.0
        config.export.alpha_min = 0.15
        # The reference DTU recipe renders on a black background.
        config.render.background = (0.0, 0.0, 0.0)
        # The reference implementation resets opacity to 0.01 and trains the
        # non-DC SH coefficients with a twentieth of the DC learning rate.
        config.model.opacity_reset_max = 0.01
        config.model.paper_sh_lr_split = True
        # Weights of the reference DTU recipe (configs/dtu/base.yaml).
        config.loss.geometric_start = 15000
        config.loss.mono_normal = 0.01
        config.loss.consistent_normal = 0.05
        config.loss.depth_normal = 0.0
        config.loss.distortion = 1000.0
        config.loss.l1_scale = 0.5
        config.loss.cos_temperature = 0.01
        # The reference implementation does not reorient the monocular prior.
        config.loss.normal_prior_facing = "as-is"
        # The reference implementation uses the gaussian-window SSIM.
        config.loss.reference_ssim = True
        config.densify.start = 500
        config.densify.stop = 15000
        config.densify.interval = 100
        config.densify.gradient_threshold = 0.0005
        config.densify.prune_opacity = 0.005
        config.densify.reset_opacity_every = 3000
        config.densify.grow_scale_fraction = 0.01
        config.densify.prune_scale_fraction = 0.01
        # The published kernel reports the screen-space gradient in pixels and
        # the reference threshold refers to that quantity.
        config.densify.gradient_space = "pixel"
        # The reference splits a Gaussian along its largest scaling axis.
        config.densify.split_rule = "max_axis"
        # The reference additionally splits large in-scene Gaussians that are
        # visible in the sampled cameras, independent of the gradient rule.
        config.densify.large_percent_dense = 0.01
        config.densify.large_sample_cams = 30
        config.train.steps = 30000
        return config

    def losses(self, pipeline, sample, output, step):
        predicted = pipeline.model.compensate(output.rgb, sample["index"])
        target = sample["image"]
        weight = self.config.loss.ssim
        if self.config.loss.reference_ssim:
            # The reference implementation weights the L1 term against a
            # gaussian-window SSIM.
            absolute = (predicted - target).abs().mean()
            terms = {
                "photometric": (1 - weight) * absolute
                + weight * reference_ssim_loss(predicted, target)
            }
        else:
            terms = {
                "photometric": photometric_loss(predicted, target, weight)
            }
        if self.config.loss.l1_scale > 0:
            # The reference pulls the smallest scale axis of every in-scene
            # Gaussian towards zero; the gradient must reach the parameters.
            smallest = pipeline.model.scales.amin(-1)
            inside = pipeline.inside_mask()
            if bool(inside.any()):
                terms["l1_scale"] = (
                    self.config.loss.l1_scale * smallest[inside].mean()
                )
            else:
                terms["l1_scale"] = self.config.loss.l1_scale * smallest.mean()
        return terms

    def regularization(self, pipeline, sample, output, step):
        settings = self.config.loss
        terms = {}
        render_normal = F.normalize(output.normals, dim=-1)
        prior = None
        if settings.mono_normal > 0 and step >= settings.mono_normal_start:
            prior = self._prior_normal(pipeline, sample)
            if prior is not None:
                terms["mono_normal"] = settings.mono_normal * normal_l1_cosine(
                    render_normal, prior
                )
        needs_depth = (
            settings.consistent_normal > 0
            and step >= settings.consistent_normal_start
        ) or (
            settings.depth_normal > 0 and step >= settings.depth_normal_start
        ) or (
            settings.normal_curvature > 0
            and step >= settings.normal_curvature_start
        )
        depth_normal = None
        if needs_depth:
            # The reference depth-derived normal points AWAY from the camera; the
            # host operator returns camera-facing normals, so the sign is flipped.
            depth_normal, _ = pipeline.ops(
                "geometry.depth_normals@1",
                output.depth.squeeze(-1),
                pipeline.centred_intrinsic(sample["camera"]),
            )
            depth_normal = -depth_normal
        if (
            settings.consistent_normal > 0
            and step >= settings.consistent_normal_start
            and depth_normal is not None
        ):
            terms["consistent_normal"] = (
                settings.consistent_normal
                * normal_l1_cosine(depth_normal, render_normal)
            )
        if settings.depth_normal > 0 and step >= settings.depth_normal_start:
            if prior is None:
                prior = self._prior_normal(pipeline, sample)
            if prior is not None and depth_normal is not None:
                confidence = cosine_confidence(
                    render_normal, prior, settings.cos_temperature
                )
                terms["depth_normal"] = settings.depth_normal * normal_l1_cosine(
                    depth_normal, prior, confidence
                )
        if (
            settings.normal_curvature > 0
            and step >= settings.normal_curvature_start
            and depth_normal is not None
        ):
            terms["normal_curvature"] = (
                settings.normal_curvature * normal_curvature(depth_normal).mean()
            )
        if (
            settings.distortion > 0
            and step >= settings.distortion_start
            and output.distortion_l2 is not None
        ):
            distortion = output.distortion_l2.squeeze(-1)
            weight = edge_aware_weight(sample["image"])
            terms["distortion"] = settings.distortion * (distortion * weight).mean()
        return terms

    def _prior_normal(self, pipeline, sample):
        """Monocular normal prior, optionally reoriented towards the camera."""
        if "prior_normal" not in sample:
            raise ValueError("VCR-GauS normal supervision requires a normal prior for each training image")
        prior = sample["prior_normal"]
        if prior.shape[-1] != 3:
            raise ValueError("VCR-GauS normal priors must have three channels")
        prior = F.normalize(prior, dim=-1)
        if self.config.loss.normal_prior_facing == "camera":
            rays = pipeline.ops(
                "camera.pixel_rays@1",
                pipeline.centred_intrinsic(sample["camera"]),
                width=sample["camera"].width,
                height=sample["camera"].height,
            )
            agreement = (prior * rays).sum(-1)
            finite = torch.isfinite(agreement)
            if bool(finite.any()):
                if float(torch.median(agreement[finite])) > 0:
                    prior = -prior
        return prior

    def photometric_only(self, step):
        return step < self.config.loss.geometric_start

    @classmethod
    def make_pipeline(cls, config):
        from ..vcr_pipeline import VCRPipeline

        if torch.device(config.train.device).type != "cuda":
            raise ValueError("VCR-GauS requires a CUDA device")
        return VCRPipeline(config, cls(config))
