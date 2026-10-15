"""Explicit native-weighted RaDe-GS L2 moment backend.

Unlike the Torch reference, this backend uses the pinned port's rasterization
support, alpha threshold and early termination for ALL outputs. L2 uses the
same per-pixel native ray-depth planes and detached native weights. L1 is not
provided: the upstream center-depth L1 helper is a different quantity.
"""

from __future__ import annotations

import inspect
import math

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..studio.renderers import RenderOutput


class GsplatRadeMomentRenderer:
    def __init__(self, config, *, ops, stable_geometry=False):
        from gsplat import rasterization
        from gsplat.cuda._wrapper import (
            fully_fused_projection,
            isect_offset_encode,
            isect_tiles,
            rasterize_to_pixels,
        )

        if "radegs" not in inspect.signature(fully_fused_projection).parameters:
            raise RuntimeError("the pinned gsplat RaDe-GS projection API is required")
        self.project = fully_fused_projection
        self.intersections = isect_tiles
        self.offsets = isect_offset_encode
        self.pixels = rasterize_to_pixels
        self.rasterization = rasterization
        self.config, self.ops = config, ops
        self.stable_geometry = stable_geometry

    def geometry(self, scene, camera):
        result = self.project(
            scene.means,
            None,
            scene.quats,
            scene.scales,
            camera.world_to_camera[None],
            camera.K[None],
            camera.width,
            camera.height,
            eps2d=self.config.eps2d,
            near_plane=self.config.near,
            far_plane=self.config.far,
            packed=False,
            calc_compensations=self.config.antialiased,
            opacities=scene.opacities,
            radegs=not self.stable_geometry,
        )
        if len(result) != (5 if self.stable_geometry else 8):
            raise RuntimeError("unexpected pinned RaDe-GS projection tuple")
        if self.stable_geometry:
            radii, xy, z, conics, compensation, *_ = result
            indices = (radii[0] > 0).all(-1).nonzero().flatten()
            rows = []

            def geometry_chunk(means, quats, scales, transform, intrinsics):
                return self.ops(
                    "gaussian.regularized_ray_geometry@1",
                    means,
                    quats,
                    scales,
                    transform,
                    intrinsics,
                    width=camera.width,
                    height=camera.height,
                )

            for batch in indices.split(16384):
                operands = (
                    scene.means[batch],
                    scene.quats[batch],
                    scene.scales[batch],
                    camera.world_to_camera,
                    camera.K,
                )
                rows.append(
                    checkpoint(
                        geometry_chunk,
                        *operands,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                    if torch.is_grad_enabled()
                    else geometry_chunk(*operands)
                )
            values = []
            for column, shape in enumerate(
                ((len(scene.means),), (len(scene.means), 2), (len(scene.means), 3))
            ):
                base = scene.means.new_zeros(shape)
                if rows:
                    base = base.index_copy(
                        0, indices, torch.cat([row[column] for row in rows])
                    )
                values.append(base[None])
            return radii, xy, z, conics, compensation, *values
        return result

    def render(self, scene, camera, *, selected=None):
        if selected is not None:
            raise ValueError(
                "native moment RaDe-GS does not implement an ambiguity selection mask"
            )
        if scene.means.device.type != "cuda" or scene.means.dtype != torch.float32:
            raise ValueError("native moment RaDe-GS requires explicit CUDA float32")
        radii, xy, z, conics, compensation, ray_t, planes, normals = self.geometry(
            scene, camera
        )
        if xy.requires_grad:
            xy.retain_grad()
        opacity = scene.opacities[None]
        if compensation is not None:
            opacity = opacity * compensation
        colors = self.ops(
            "gaussian.sh_color@1",
            scene.sh,
            scene.means - camera.origin,
            degree=scene.sh_degree,
        )[None]
        tile = self.config.tile_size
        tw, th = math.ceil(camera.width / tile), math.ceil(camera.height / tile)
        _, intersections, ids = self.intersections(
            xy, radii, z, tile, tw, th, packed=False
        )
        offsets = self.offsets(intersections, 1, tw, th)
        rgb, alpha, znum, median, nnum, _ = self.pixels(
            xy,
            conics,
            colors,
            opacity,
            ray_t,
            planes,
            normals,
            camera.width,
            camera.height,
            tile,
            offsets,
            ids,
            backgrounds=colors.new_tensor([self.config.background]),
            Ks=camera.K[None],
            packed=False,
            absgrad=True,
            radegs=True,
            distloss=False,
        )
        valid = (radii > 0).all(-1)
        safe_xy = torch.where(valid[..., None], xy, 0)
        safe_t = torch.where(valid, ray_t, 0)
        safe_planes = torch.where(valid[..., None], planes, 0)
        center = (
            safe_t[valid].detach().median() if valid.any() else safe_t.new_zeros(())
        )
        a = safe_t - center + ((safe_xy - camera.K[:2, 2]) * safe_planes).sum(-1)
        bx, by = (-safe_planes).unbind(-1)
        features = torch.stack(
            (
                a,
                bx,
                by,
                a.square(),
                2 * a * bx,
                2 * a * by,
                bx.square(),
                2 * bx * by,
                by.square(),
            ),
            -1,
        )
        features = torch.where(valid[..., None], features, 0)
        moments, frozen_alpha, *_ = self.pixels(
            xy.detach(),
            conics.detach(),
            features,
            opacity.detach(),
            None,
            None,
            None,
            camera.width,
            camera.height,
            tile,
            offsets,
            ids,
            packed=False,
            absgrad=False,
            radegs=False,
            distloss=False,
        )
        rays = self.ops(
            "camera.pixel_rays@1", camera.K, width=camera.width, height=camera.height
        )
        pixels = (rays @ camera.K.T)[..., :2] - camera.K[:2, 2]
        x, y = pixels.unbind(-1)
        m = moments[0]
        first = m[..., 0] + m[..., 1] * x + m[..., 2] * y
        second = (
            m[..., 3]
            + m[..., 4] * x
            + m[..., 5] * y
            + m[..., 6] * x.square()
            + m[..., 7] * x * y
            + m[..., 8] * y.square()
        )
        variance = (
            2
            * (frozen_alpha[0, ..., 0].detach() * second - first.square())
            / rays.square().sum(-1)
        ).clamp_min(0)[..., None]
        return RenderOutput(
            rgb[0],
            alpha[0],
            znum[0] / alpha[0].clamp_min(1e-8),
            F.normalize(nnum[0], dim=-1),
            None,
            variance,
            median[0],
            torch.zeros_like(alpha[0]),
            torch.zeros_like(alpha[0]),
            xy,
            radii[0].amax(-1),
            normal_numerator=nnum[0],
            distortion_profile="l2",
        )

    def render_rgb(self, scene, camera, *, selected=None):
        """Same native RGB support/VJP, without unused geometry or L2 passes."""
        if selected is not None:
            raise ValueError("native moment RGB path does not implement selection")
        if scene.means.device.type != "cuda" or scene.means.dtype != torch.float32:
            raise ValueError("native moment RGB path requires explicit CUDA float32")
        colors = self.ops(
            "gaussian.sh_color@1",
            scene.sh,
            scene.means - camera.origin,
            degree=scene.sh_degree,
        )
        rgb, alpha, meta = self.rasterization(
            scene.means,
            scene.quats,
            scene.scales,
            scene.opacities,
            colors,
            camera.world_to_camera[None],
            camera.K[None],
            camera.width,
            camera.height,
            packed=False,
            render_mode="RGB",
            absgrad=True,
            RadeGS=False,
            distloss=False,
            eps2d=self.config.eps2d,
            near_plane=self.config.near,
            far_plane=self.config.far,
            tile_size=self.config.tile_size,
            backgrounds=colors.new_tensor([self.config.background]),
            rasterize_mode="antialiased" if self.config.antialiased else "classic",
        )
        xy = meta["means2d"]
        if xy.requires_grad:
            xy.retain_grad()
        zero = torch.zeros_like(alpha[0])
        return RenderOutput(
            rgb[0],
            alpha[0],
            zero,
            zero.expand(-1, -1, 3),
            None,
            zero,
            zero,
            zero,
            zero,
            xy,
            meta["radii"][0].amax(-1),
            distortion_profile="none",
        )
