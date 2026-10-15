"""Direct public-API adapter for the Apache-2.0 gsplat RaDe-GS port.

Dependency: Mollylulu/gsplat_radegs@23b0f82551a938a05b5b99df06edf220b9305a51.
The port's auxiliary distortion uses center-depth L1, whereas the original
RaDe-GS contract uses per-pixel L2. Obtain the canonical loss from the explicit
Torch composition rather than silently equating these different semantics.
"""

from __future__ import annotations

import inspect

import torch

from ..studio.batched_renderer import BatchedTorchRenderer
from ..studio.renderers import RenderOutput


class GsplatRadeRenderer:
    def __init__(self, config, *, ops):
        self.config = config
        try:
            from gsplat import rasterization
        except ImportError as exc:
            raise RuntimeError(
                "Install the pinned optional dependency with pip install 'splatforge[gsplat-radegs]'; an ordinary gsplat wheel does not implement RadeGS=True"
            ) from exc
        if "RadeGS" not in inspect.signature(rasterization).parameters:
            raise RuntimeError(
                "This gsplat build lacks RadeGS=True. Use Mollylulu/gsplat_radegs@23b0f82551a938a05b5b99df06edf220b9305a51 in an isolated environment."
            )
        self.rasterization = rasterization
        self.reference = BatchedTorchRenderer(config, mode="radegs", ops=ops)

    def render(self, scene, camera, *, selected=None):
        if scene.means.device.type != "cuda" or scene.means.dtype != torch.float32:
            raise ValueError("gsplat-radegs requires explicit CUDA float32 tensors")
        rendered, alpha, meta = self.rasterization(
            scene.means,
            scene.quats,
            scene.scales,
            scene.opacities,
            scene.sh,
            camera.world_to_camera[None],
            camera.K[None],
            camera.width,
            camera.height,
            sh_degree=scene.sh_degree,
            RadeGS=True,
            packed=False,
            distloss=True,
            render_mode="RGB+D",
            absgrad=True,
            eps2d=self.config.eps2d,
            near_plane=self.config.near,
            far_plane=self.config.far,
            tile_size=self.config.tile_size,
            backgrounds=scene.means.new_tensor([self.config.background]),
            rasterize_mode="antialiased" if self.config.antialiased else "classic",
        )
        required = ("expected_depths", "expected_normals", "median_depths")
        if any(key not in meta for key in required):
            raise RuntimeError(
                "RaDe-GS provider did not return its required geometry outputs"
            )
        # Canonical loss has a distinct derivation from the upstream helper.
        auxiliary = self.reference.render(scene, camera, selected=selected)
        projected = meta["means2d"]
        if projected.requires_grad:
            projected.retain_grad()
        return RenderOutput(
            rendered[0, ..., :3],
            alpha[0],
            meta["expected_depths"][0],
            meta["expected_normals"][0],
            auxiliary.distortion_l1,
            auxiliary.distortion_l2,
            meta["median_depths"][0],
            auxiliary.ray_color,
            auxiliary.ambiguity_mask,
            projected,
            auxiliary.projected_radii,
        )
