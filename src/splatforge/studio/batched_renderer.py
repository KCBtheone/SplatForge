"""Exact eager tile equations, batched and checkpointed to bound saved tensors.

Padding has zero opacity. Sorting remains per pixel for L1 distortion. The
batch/event budget controls execution memory, never input or Gaussian count.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..gaussian import SurfaceOperators
from ..providers.torch.gaussian.reference import rotation_matrix
from .renderers import RenderOutput, project_geometry
from .tile_binning import tile_members


class BatchedTorchRenderer:
    def __init__(self, config, *, mode, trunc_sigma=2.0, ops=None, compiled=False):
        self.config = config
        self.mode = mode
        self.trunc_sigma = trunc_sigma
        self.ops = ops or SurfaceOperators()
        self.compiled = compiled
        self.compiled_kernel = None
        if compiled:
            from ..providers.torch.gaussian.tile import checkpointed_tile_batch

            self.compiled_kernel = torch.compile(
                checkpointed_tile_batch, dynamic=True, fullgraph=True
            )

    def render_training(self, scene, camera, *, selected=None, geometric=True):
        requested = (
            "none"
            if self.mode == "ambisur" or not geometric
            else ("l1" if self.mode == "gof" else "l2")
        )
        return self.render(scene, camera, selected=selected, distortions=requested)

    def render(self, scene, camera, *, selected=None, distortions="both"):
        if self.compiled and scene.means.device.type != "cuda":
            raise ValueError(
                "torch-compiled explicitly requires CUDA; no CPU/eager fallback"
            )
        ops, config = self.ops, self.config
        means, _, cov_camera, J, means2d, screen_cov, filtered = project_geometry(
            scene, camera, ops, config.eps2d, config.near
        )
        if means2d.requires_grad:
            means2d.retain_grad()
            means2d.absgrad = torch.zeros_like(means2d)
        inv_screen = torch.linalg.inv(filtered)
        extent = self.trunc_sigma if self.mode == "ambisur" else config.sigma_extent
        radii = extent * torch.linalg.eigvalsh(filtered)[:, -1].clamp_min(0).sqrt()
        opacity = scene.opacities
        if config.antialiased:
            opacity = (
                opacity
                * (
                    torch.linalg.det(screen_cov).clamp_min(0)
                    / torch.linalg.det(filtered).clamp_min(1e-12)
                ).sqrt()
            )
        colors = ops(
            "gaussian.sh_color@1",
            scene.sh,
            scene.means - camera.origin,
            degree=scene.sh_degree,
        )
        ray_colors = ops(
            "gaussian.sh_color@1",
            scene.sh,
            scene.means.detach() - camera.origin,
            degree=scene.sh_degree,
        )
        rays = ops(
            "camera.pixel_rays@1", camera.K, width=camera.width, height=camera.height
        )
        normals = means.new_zeros((len(means), 3))
        gain = means.new_zeros((len(means), 2))
        inverse_cov = torch.eye(3, device=means.device, dtype=means.dtype).expand(
            len(means), 3, 3
        )
        distance = means.new_zeros(len(means))
        if self.mode == "radegs":
            fullJ = torch.cat((J, F.normalize(means, dim=-1)[:, None]), 1)
            transformed = fullJ @ cov_camera @ fullJ.transpose(-1, -2)
            gain = (
                transformed[:, 2:3, :2] @ torch.linalg.inv(transformed[:, :2, :2])
            ).squeeze(1)
            ray_normal = torch.cat((gain, -torch.ones_like(gain[:, :1])), 1)
            normals = F.normalize(
                (fullJ.transpose(-1, -2) @ ray_normal.unsqueeze(-1)).squeeze(-1), dim=-1
            )
        elif self.mode == "ambisur":
            axis = scene.scales.detach().argmin(-1)
            world = (
                rotation_matrix(scene.quats)
                .gather(2, axis[:, None, None].expand(-1, 3, 1))
                .squeeze(-1)
            )
            normals = world @ camera.world_to_camera[:3, :3].T
            normals = torch.where(
                (normals * means).sum(-1, keepdim=True) > 0, -normals, normals
            )
            distance = (normals * means).sum(-1)
        elif self.mode == "gof":
            inverse_cov = torch.linalg.inv(cov_camera)
        else:
            raise ValueError(f"unsupported batched rendering mode: {self.mode}")
        selection = (
            selected
            if selected is not None
            else torch.zeros(len(means), dtype=torch.bool, device=means.device)
        )
        bounds = (
            torch.cat(
                (means2d.detach(), radii.detach()[:, None], means.detach()[:, 2:3]), 1
            )
            .cpu()
            .numpy()
        )
        groups = defaultdict(list)
        layout = []
        tile_size = config.tile_size
        binned = tile_members(
            bounds,
            width=camera.width,
            height=camera.height,
            tile_size=tile_size,
            near=config.near,
            far=config.far,
        )
        for top in range(0, camera.height, tile_size):
            for left in range(0, camera.width, tile_size):
                height, width = (
                    min(tile_size, camera.height - top),
                    min(tile_size, camera.width - left),
                )
                index = len(layout)
                ids = binned[index]
                layout.append((top, left, height, width))
                groups[(height, width, max(1, (len(ids) - 1).bit_length()))].append(
                    (index, ids)
                )
        tiles = [None] * len(layout)
        focal = torch.stack((camera.K[0, 0], camera.K[1, 1]))

        def compose(alpha, color, depth, normal):
            return ops(
                "splat.alpha_composite@3",
                alpha,
                color,
                depth,
                normal,
                distortions=distortions,
            )

        batch_composite = compose

        def batch_ray_color(w, c, r):
            return ops("gaussian.ray_color_variance@2", w, c, r)

        def render_batch(
            ids,
            valid_slots,
            tile_rays,
            pixels,
            xy,
            q2,
            color,
            ray_color,
            opacities,
            local_means,
            normal,
            gains,
            q3,
            plane_distance,
        ):
            delta = pixels[:, None] - xy[ids][:, :, None, :]
            if self.compiled:
                proxy_xy = (xy[ids] - camera.K[:2, 2]) * local_means[ids, 2:3] / focal
                proxy = torch.cat((proxy_xy, local_means[ids, 2:3]), -1)[
                    :, :, None
                ].expand(-1, -1, tile_rays.shape[1], -1)
                if self.mode == "gof" and proxy.requires_grad:

                    def accumulate_compiled(
                        gradient, indices=ids, gate=valid_slots, target=means2d
                    ):
                        value = (
                            (
                                gradient[..., :2]
                                * local_means[indices, 2].detach()[:, :, None, None]
                                / focal
                            )
                            .abs()
                            .sum(2)
                        )
                        target.absgrad.index_add_(0, indices[gate], value[gate])

                    proxy.register_hook(accumulate_compiled)
                elif delta.requires_grad:

                    def accumulate_compiled(
                        gradient, indices=ids, gate=valid_slots, target=means2d
                    ):
                        target.absgrad.index_add_(
                            0, indices[gate], gradient.abs().sum(2)[gate]
                        )

                    delta.register_hook(accumulate_compiled)
                return self.compiled_kernel(
                    delta,
                    proxy,
                    q2[ids],
                    color[ids],
                    ray_color[ids],
                    opacities[ids],
                    local_means[ids],
                    normal[ids],
                    gains[ids],
                    q3[ids],
                    plane_distance[ids],
                    valid_slots,
                    selection[ids],
                    tile_rays,
                    color.new_tensor(config.background),
                    mode=self.mode,
                    near=config.near,
                    far=config.far,
                    extent=extent,
                    distortions=distortions,
                )
            squared = torch.einsum("tlpi,tlij,tlpj->tlp", delta, q2[ids], delta)
            if self.mode == "gof":
                proxy_xy = (xy[ids] - camera.K[:2, 2]) * local_means[ids, 2:3] / focal
                proxy = torch.cat((proxy_xy, local_means[ids, 2:3]), -1)[
                    :, :, None
                ].expand(-1, -1, tile_rays.shape[1], -1)
                if proxy.requires_grad:

                    def absolute_grad(
                        gradient, indices=ids, gate=valid_slots, target=xy
                    ):
                        value = (
                            (
                                gradient[..., :2]
                                * local_means[indices, 2].detach()[:, :, None, None]
                                / focal
                            )
                            .abs()
                            .sum(2)
                        )
                        target.absgrad.index_add_(0, indices[gate], value[gate])

                    proxy.register_hook(absolute_grad)
                qd = torch.einsum("tlij,tpj->tlpi", q3[ids], tile_rays)
                denominator = (qd * tile_rays[:, None]).sum(-1).clamp_min(1e-12)
                depth = (proxy * qd).sum(-1) / denominator
                offset = depth[..., None] * tile_rays[:, None] - proxy
                exponent = torch.einsum("tlpi,tlij,tlpj->tlp", offset, q3[ids], offset)
                response = torch.exp(-0.5 * exponent.clamp_min(0))
                pixel_normals = F.normalize(-qd, dim=-1)
            else:
                if delta.requires_grad:

                    def absolute_grad(
                        gradient, indices=ids, gate=valid_slots, target=xy
                    ):
                        target.absgrad.index_add_(
                            0, indices[gate], gradient.abs().sum(2)[gate]
                        )

                    delta.register_hook(absolute_grad)
                response = torch.exp(-0.5 * squared)
                pixel_normals = normal[ids][:, :, None, :].expand(
                    -1, -1, tile_rays.shape[1], -1
                )
                if self.mode == "radegs":
                    depth = (
                        torch.linalg.vector_norm(local_means[ids], dim=-1)[:, :, None]
                        + (gains[ids][:, :, None, :] * delta).sum(-1)
                    ) / torch.linalg.vector_norm(tile_rays, dim=-1)[:, None]
                else:
                    denominator = (pixel_normals * tile_rays[:, None]).sum(-1)
                    safe = torch.where(
                        denominator.abs() > 1e-8,
                        denominator,
                        torch.ones_like(denominator),
                    )
                    depth = plane_distance[ids][:, :, None] / safe
                    response = torch.where(denominator.abs() > 1e-8, response, 0)
            response = torch.where(
                valid_slots[:, :, None]
                & (squared <= extent**2)
                & (depth > config.near)
                & (depth < config.far),
                response,
                0,
            )
            rgb, alpha, znum, nnum, l1, l2, median, weights = batch_composite(
                opacities[ids][:, :, None] * response, color[ids], depth, pixel_normals
            )
            expected = znum / alpha.clamp_min(1e-8)
            if self.mode == "ambisur":
                dnum = (weights * plane_distance[ids][:, :, None]).sum(1)
                denominator = (nnum * tile_rays).sum(-1)
                safe = torch.where(
                    denominator.abs() > 1e-8, denominator, torch.ones_like(denominator)
                )
                expected = torch.where(denominator.abs() > 1e-8, dnum / safe, 0)
            expected = torch.where(alpha > 1e-8, expected, 0)
            variance = (
                batch_ray_color(weights, ray_color[ids], rgb)
                if self.mode == "ambisur"
                else torch.zeros_like(alpha)
            )
            ambiguity = (weights * selection[ids][:, :, None].to(weights.dtype)).sum(1)
            return torch.cat(
                (
                    rgb + (1 - alpha[..., None]) * rgb.new_tensor(config.background),
                    alpha[..., None],
                    expected[..., None],
                    F.normalize(nnum, dim=-1),
                    l1[..., None],
                    l2[..., None],
                    median[..., None],
                    variance[..., None],
                    ambiguity[..., None],
                ),
                -1,
            )

        for (height, width, bucket), members in groups.items():
            maximum = (
                (1 << bucket)
                if self.compiled
                else max(1, max(len(ids) for _, ids in members))
            )
            batch_size = max(
                1,
                min(
                    config.tile_batch_size,
                    config.tile_event_budget // (maximum * height * width),
                ),
            )
            for start in range(0, len(members), batch_size):
                batch = members[start : start + batch_size]
                indices = np.zeros((len(batch), maximum), np.int64)
                gates = np.zeros_like(indices, dtype=bool)
                ray_rows = []
                for row, (tile_index, ids) in enumerate(batch):
                    indices[row, : len(ids)], gates[row, : len(ids)] = ids, True
                    top, left, _, _ = layout[tile_index]
                    ray_rows.append(
                        rays[top : top + height, left : left + width].reshape(-1, 3)
                    )
                ids = torch.as_tensor(indices, device=means.device)
                valid = torch.as_tensor(gates, device=means.device)
                tile_rays = torch.stack(ray_rows)
                pixels = (tile_rays @ camera.K.T)[..., :2]
                operands = (
                    ids,
                    valid,
                    tile_rays,
                    pixels,
                    means2d,
                    inv_screen,
                    colors,
                    ray_colors,
                    opacity,
                    means,
                    normals,
                    gain,
                    inverse_cov,
                    distance,
                )
                output = (
                    # Tile equations are deterministic. RNG preservation would
                    # enter CUDA RNG-state cloning from AOTAutograd's fake-tensor
                    # backward on Torch 2.1, even though this function uses no RNG.
                    checkpoint(
                        render_batch,
                        *operands,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                    if torch.is_grad_enabled() and not self.compiled
                    else render_batch(*operands)
                )
                for row, (tile_index, _) in enumerate(batch):
                    tiles[tile_index] = output[row].reshape(height, width, -1)
        rows = []
        columns = (camera.width + tile_size - 1) // tile_size
        for start in range(0, len(tiles), columns):
            rows.append(torch.cat(tiles[start : start + columns], 1))
        combined = torch.cat(rows, 0)
        channels = torch.split(combined, (3, 1, 1, 3, 1, 1, 1, 1, 1), dim=-1)
        return RenderOutput(
            *channels,
            projected_means=means2d,
            projected_radii=radii,
            distortion_profile=distortions,
        )
