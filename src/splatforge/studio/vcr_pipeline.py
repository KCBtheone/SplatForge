"""VCR-GauS scene bounds, large-point splits and importance pruning."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..providers.torch.gaussian.reference import rotation_matrix
from .cameras import Camera
from .densification import GaussianDensifier
from .pipeline import GaussianPipeline
from .renderers import GaussianView
from .vcr_renderer import centred_intrinsic


class VCRDensifier(GaussianDensifier):
    def split_children(self, model, split):
        if self.config.split_rule == "random":
            return super().split_children(model, split)
        return self.axis_children(model, split)

    def axis_children(self, model, split):
        scales = model.scales.detach()[split]
        largest, axis = scales.max(-1)
        rotation = rotation_matrix(model.quats.detach()[split])
        direction = rotation.gather(2, axis[:, None, None].expand(-1, 3, 1)).squeeze(-1)
        offset = direction * (2 * largest[:, None] / 3)
        centers = model.means.detach()[split]
        means = torch.stack((centers + offset, centers - offset), dim=1).reshape(-1, 3)
        child_scales = scales.clone()
        rows = torch.arange(len(split), device=model.device)
        child_scales[rows, axis] /= 1.6
        return split.repeat_interleave(2), means, child_scales.log().repeat_interleave(2, dim=0)

    def prune_mask(self, model, optimizer, keep):
        indices = keep.nonzero().squeeze(-1)
        self._replace(model, optimizer, indices, indices.new_empty(0))


class VCRPipeline(GaussianPipeline):
    densifier_type = VCRDensifier

    def __init__(self, config, method):
        super().__init__(config, method)
        self._inside_trans = None
        self._inside_scale = None
        self._init_inside_bounds()

    def _init_inside_bounds(self):
        """Scene bounds used by the reference inside mask.

        The reference implementation stores them in ``meta.json`` next to the
        scene and otherwise derives them from the sparse point cloud.
        """
        root = Path(self.config.data.source)
        meta = root / "meta.json"
        trans = scale = None
        if meta.is_file():
            try:
                payload = json.loads(meta.read_text())
                trans = np.asarray(payload["trans"], dtype=np.float64)
                scale = np.asarray(payload["scale"], dtype=np.float64)
            except (OSError, ValueError, KeyError):
                trans = scale = None
        if trans is None or scale is None:
            points = np.asarray(self.data.points, dtype=np.float64)
            trans = points.mean(axis=0)
            scale = np.abs(points).max(axis=0) * 1.1
        device = self.model.device
        self._inside_trans = torch.as_tensor(
            trans, device=device, dtype=torch.float32
        )
        self._inside_scale = torch.as_tensor(
            scale, device=device, dtype=torch.float32
        ).clamp_min(1e-12)

    def inside_mask(self, points=None):
        """Points inside the normalised scene box, as the reference defines it."""
        if points is None:
            points = self.model.means.detach()
        normalized = (points - self._inside_trans) / self._inside_scale
        return normalized.abs().amax(-1) < 1.5

    def centred_intrinsic(self, camera):
        """Centred intrinsic at the training resolution."""
        return centred_intrinsic(camera)

    def _frame_camera(self, index):
        """Training-resolution camera without loading the image."""
        frame = self.data.frames[index]
        camera = frame.camera
        downscale = max(int(self.config.data.downscale), 1)
        width = max(1, camera.width // downscale)
        height = max(1, camera.height // downscale)
        intrinsic = camera.K.clone()
        intrinsic[0] *= width / camera.width
        intrinsic[1] *= height / camera.height
        return Camera(
            intrinsic,
            camera.world_to_camera,
            width,
            height,
            camera.name,
        ).to(self.model.device)

    def _sample_cameras(self, count):
        """Deterministic camera subset used by the large-point branch."""
        indices = list(self.data.train_indices)
        if count >= len(indices):
            return indices
        stride = max(len(indices) // count, 1)
        return indices[::stride][:count]

    @torch.no_grad()
    def _large_point_split(self, step, optimizer):
        settings = self.config.densify
        if settings.large_percent_dense <= 0:
            return
        if step < settings.start or step >= settings.stop:
            return
        if step % settings.interval:
            return
        device = self.model.device
        if device.type == "cuda":
            reserved = torch.cuda.memory_reserved(device) / 1024**3
            if reserved >= settings.large_max_memory_gib:
                return
        scales = self.model.scales.detach()
        largest = scales.amax(-1)
        big = largest > settings.large_percent_dense * self.densifier.extent
        inside = self.inside_mask()
        candidates = (big & inside).nonzero().squeeze(-1)
        if not len(candidates):
            return
        view = GaussianView.from_model(self.model)
        visible = torch.zeros_like(big)
        for index in self._sample_cameras(settings.large_sample_cams):
            camera = self._frame_camera(index)
            counts = self.renderer.render_importance(view, camera)
            visible |= counts.detach() > 0
        parents = candidates[visible[candidates]]
        if not len(parents):
            return
        room = settings.max_gaussians - self.model.count
        if room <= 0:
            return
        if len(parents) > room:
            parents = parents[:room]
        children, means, log_scales = self.densifier.axis_children(self.model, parents)
        keep = torch.ones_like(big)
        keep[parents] = False
        self.densifier._replace(
            self.model, optimizer, keep.nonzero().squeeze(-1), children,
            child_means=means, child_log_scales=log_scales,
        )

    @torch.no_grad()
    def _importance_prune(self, step, optimizer):
        settings = self.config.loss
        if step not in tuple(settings.prune_iterations):
            return
        view = GaussianView.from_model(self.model)
        importance = torch.zeros(
            self.model.count, device=self.model.device, dtype=torch.float32
        )
        for index in self.data.train_indices:
            camera = self._frame_camera(index)
            importance += self.renderer.render_importance(view, camera).to(
                importance.dtype
            )
        volume = self.model.scales.detach().prod(-1)
        if not bool(torch.isfinite(volume).all()) or float(volume.max()) <= 0:
            return
        kth = torch.sort(volume, descending=True).values[
            min(int(len(volume) * 0.9), len(volume) - 1)
        ]
        score = (volume / kth.clamp_min(1e-12)).pow(settings.prune_v_pow) * (
            importance
        )
        occurrence = tuple(settings.prune_iterations).index(step)
        fraction = settings.prune_percent * (
            settings.prune_decay ** occurrence
        )
        keep = int(len(score) * max(0.0, 1.0 - fraction))
        if keep <= 0 or keep >= len(score):
            return
        order = torch.argsort(score, descending=True)[:keep]
        keep_mask = torch.zeros(len(score), dtype=torch.bool, device=score.device)
        keep_mask[order] = True
        self.densifier.prune_mask(self.model, optimizer, keep_mask)

    def after_step(self, step, optimizer):
        super().after_step(step, optimizer)
        if not self.config.densify.enabled:
            return
        self._large_point_split(step, optimizer)
        self._importance_prune(step, optimizer)
