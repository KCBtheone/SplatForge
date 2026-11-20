"""Exact eager tile equations, batched and checkpointed to bound saved tensors.

Padding has zero opacity. Sorting remains per pixel for L1 distortion. The
batch/event budget controls execution memory, never input or Gaussian count.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..gaussian import SurfaceOperators
from ..providers.torch.gaussian.reference import rotation_matrix
from .renderers import RenderOutput, project_geometry
from .tile_binning import tile_members


def _accumulate_absolute_gradient(target, ids, valid, value):
    """Fixed-size scatter; padded slots add zero without CUDA nonzero/sync."""
    target.index_add_(
        0,
        ids.reshape(-1),
        torch.where(valid[..., None], value, 0).reshape(-1, 2),
    )


def _pack_cpu_memberships(groups, device):
    """Fresh per-render membership upload; no model/view cache."""
    total = sum(max(1, max(len(ids) for _, ids in members))*len(members)
                for members in groups.values())
    payload = np.full(total, -1, dtype=np.int64)
    offset = 0
    for members in groups.values():
        maximum = max(1, max(len(ids) for _, ids in members))
        rows = payload[offset:offset+len(members)*maximum].reshape(len(members), maximum)
        for row, (_, ids) in enumerate(members):
            rows[row, :len(ids)] = ids
        offset += len(members)*maximum
    uploaded = torch.as_tensor(payload, device=device)
    return uploaded.clamp_min(0), uploaded >= 0


class BatchedTorchRenderer:
    def __init__(
        self,
        config,
        *,
        mode,
        trunc_sigma=2.0,
        ops=None,
        compiled=False,
        cuda_binning=False,
    ):
        self.config = config
        self.mode = mode
        self.trunc_sigma = trunc_sigma
        self.ops = ops or SurfaceOperators()
        self.compiled = compiled
        self.cuda_binning = cuda_binning
        self.cuda_events = False
        self.packed_membership = False  # Experimental; explicit activation only.
        self.batch_streams = 0  # Explicit GOF execution experiment, never fallback.
        self.ambisur_event_producer = None  # Set by the AmbiSuR factory or a bound probe.
        self.native_radegs_tile = None  # Explicit complete native tile execution.
        self.native_primal_pullback = None  # Original Tensor primal, input-owned adjoint.
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

    def _render_cuda_batch(
        self,
        ids,
        valid,
        rays,
        pixels,
        xy,
        q2,
        color,
        ray_color,
        opacity,
        means,
        normal,
        q3,
        distance,
        selection,
        camera,
        distortions,
        background,
    ):
        from ..providers.torch.gaussian.events_cuda import surface_events_cuda

        delta = pixels[:, None] - xy[ids][:, :, None, :]
        if self.mode == "gof":
            focal = torch.stack((camera.K[0, 0], camera.K[1, 1]))
            proxy_xy = (xy[ids] - camera.K[:2, 2]) * means[ids, 2:3] / focal
            proxy = torch.cat((proxy_xy, means[ids, 2:3]), -1)[:, :, None].expand(
                -1, -1, rays.shape[1], -1
            )
            if proxy.requires_grad:

                def absolute_grad(gradient):
                    value = (
                        (
                            gradient[..., :2]
                            * means[ids, 2].detach()[:, :, None, None]
                            / focal
                        )
                        .abs()
                        .sum(2)
                    )
                    # Boolean CUDA indexing calls nonzero and synchronizes the
                    # host for its variable-sized result. Padded slots already
                    # have a valid id; add exact zeros instead of compacting.
                    _accumulate_absolute_gradient(xy.absgrad, ids, valid, value)

                proxy.register_hook(absolute_grad)
        else:
            proxy = means[ids][:, :, None].expand(-1, -1, rays.shape[1], -1)
            if delta.requires_grad:

                def absolute_grad(gradient):
                    value = gradient.abs().sum(2)
                    _accumulate_absolute_gradient(xy.absgrad, ids, valid, value)

                delta.register_hook(absolute_grad)
        event_alpha, depth, pixel_normals = surface_events_cuda(
            delta,
            proxy,
            q2[ids],
            q3[ids],
            normal[ids],
            distance[ids],
            opacity[ids],
            rays,
            valid,
            mode=self.mode,
            near=self.config.near,
            far=self.config.far,
            extent=self.trunc_sigma
            if self.mode == "ambisur"
            else self.config.sigma_extent,
            # Reference has two independent q3[ids] nodes. Sharing one merges
            # their adjoints before the scatter and changes float32 reduction
            # order through highly conditioned inverse covariance matrices.
            q3_exponent=q3[ids] if self.mode == "gof" else None,
        )
        rgb, alpha, znum, nnum, l1, l2, median, weights = self.ops(
            "splat.alpha_composite@3",
            event_alpha,
            color[ids],
            depth,
            pixel_normals,
            distortions=distortions,
            detach_l1_weights=self.mode == "gof",
        )
        expected = znum / alpha.clamp_min(1e-8)
        if self.mode == "ambisur":
            numerator = (weights * distance[ids][:, :, None]).sum(1)
            denominator = (nnum * rays).sum(-1)
            safe = torch.where(
                denominator.abs() > 1e-8, denominator, torch.ones_like(denominator)
            )
            expected = torch.where(denominator.abs() > 1e-8, numerator / safe, 0)
        expected = torch.where(alpha > 1e-8, expected, 0)
        variance = (
            self.ops("gaussian.ray_color_variance@2", weights, ray_color[ids], rgb)
            if self.mode == "ambisur"
            else torch.zeros_like(alpha)
        )
        ambiguity = (weights * selection[ids][:, :, None].to(weights.dtype)).sum(1)
        return torch.cat(
            (
                rgb + (1 - alpha[..., None]) * background,
                alpha[..., None],
                expected[..., None],
                F.normalize(nnum, dim=-1),
                l1[..., None],
                l2[..., None],
                median[..., None],
                variance[..., None],
                ambiguity[..., None],
            ) + ((nnum,) if self.mode == "gof" else ()),
            -1,
        )

    def render(self, scene, camera, *, selected=None, distortions="both"):
        if self.ambisur_event_producer is not None and (
            self.mode!='ambisur' or self.compiled or self.cuda_events
        ):
            raise ValueError('sparse event producer requires inline AmbiSuR; no fallback')
        if self.batch_streams and (self.batch_streams!=4 or self.mode!='gof' or
                                  scene.means.device.type!='cuda' or self.compiled):
            raise ValueError('four-stream execution requires original CUDA GOF batches; no fallback')
        if self.packed_membership and (self.compiled or self.cuda_binning):
            raise ValueError('packed membership requires original eager CPU binning')
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
        bounds = torch.cat(
            (means2d.detach(), radii.detach()[:, None], means.detach()[:, 2:3]), 1
        )
        groups = defaultdict(list)
        layout = []
        tile_size = config.tile_size
        if self.cuda_binning:
            from ..providers.torch.gaussian.tile_binning_cuda import tile_members_cuda

            offsets, flat_ids = tile_members_cuda(
                bounds,
                width=camera.width,
                height=camera.height,
                tile_size=tile_size,
                near=config.near,
                far=config.far,
            )
            # Only O(tiles) metadata crosses the host boundary. All Gaussian
            # membership indices stay on CUDA; this backend is explicitly eager.
            offsets_host = offsets.cpu().tolist()
            binned = [
                flat_ids[start:stop]
                for start, stop in zip(offsets_host[:-1], offsets_host[1:])
            ]
        else:
            binned = tile_members(
                bounds.cpu().numpy(),
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
        # CPU->CUDA tensor constructors synchronize. These constants are shared
        # by all batches and their checkpoint recomputations, not recreated for
        # each tile batch. Gaussian membership and arithmetic order are intact.
        background = colors.new_tensor(config.background)
        chosen_tiles = (
            torch.tensor(
                [index for members in groups.values() for index, _ in members],
                device=means.device,
                dtype=torch.int64,
            )
            if self.cuda_binning
            else None
        )
        chosen_offset = 0
        packed_ids, packed_valid = (
            _pack_cpu_memberships(groups, means.device) if self.packed_membership else (None, None)
        )
        packed_offset = 0

        def compose(alpha, color, depth, normal):
            return ops(
                "splat.alpha_composite@3",
                alpha,
                color,
                depth,
                normal,
                distortions=distortions,
                detach_l1_weights=self.mode == "gof",
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
            *,
            tensor_primal=False,
        ):
            if self.native_primal_pullback is not None and not tensor_primal:
                if self.mode != 'radegs' or self.compiled or self.cuda_events:
                    raise ValueError('original primal/native pullback requires original eager grouping')
                original_arguments = (ids, valid_slots, tile_rays, pixels, xy, q2,
                    color, ray_color, opacities, local_means, normal, gains, q3, plane_distance)
                delta = pixels[:, None] - xy[ids][:, :, None, :]
                if delta.requires_grad:
                    def absolute_primal(gradient, indices=ids, gate=valid_slots, target=xy):
                        _accumulate_absolute_gradient(target.absgrad, indices, gate, gradient.abs().sum(2))
                    delta.register_hook(absolute_primal)
                return self.native_primal_pullback(
                    lambda: render_batch(*original_arguments, tensor_primal=True),
                    delta, q2[ids], color[ids], opacities[ids], local_means[ids],
                    normal[ids], gains[ids], valid_slots, selection[ids], tile_rays,
                    background, near=config.near, far=config.far, extent=extent,
                    distortions=distortions)
            if self.native_radegs_tile is not None:
                if self.mode != 'radegs' or self.compiled or self.cuda_events:
                    raise ValueError('native RaDe-GS tile requires original eager grouping; no fallback')
                delta = pixels[:, None] - xy[ids][:, :, None, :]
                if delta.requires_grad:
                    def absolute_native(gradient, indices=ids, gate=valid_slots, target=xy):
                        _accumulate_absolute_gradient(target.absgrad, indices, gate, gradient.abs().sum(2))
                    delta.register_hook(absolute_native)
                return self.native_radegs_tile(delta, q2[ids], color[ids], opacities[ids],
                    local_means[ids], normal[ids], gains[ids], valid_slots, selection[ids],
                    tile_rays, background, near=config.near, far=config.far,
                    extent=extent, distortions=distortions)
            if self.cuda_events:
                return self._render_cuda_batch(
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
                    q3,
                    plane_distance,
                    selection,
                    camera,
                    distortions,
                    background,
                )
            if self.ambisur_event_producer is not None:
                event_alpha,depth,pixel_normals=self.ambisur_event_producer(
                    ids,valid_slots,tile_rays,pixels,xy,q2,opacities,normal,
                    plane_distance,extent,config.near,config.far)
            else:
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
                        background,
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
                event_alpha = opacities[ids][:, :, None] * response
            rgb, alpha, znum, nnum, l1, l2, median, weights = batch_composite(
                event_alpha, color[ids], depth, pixel_normals
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
                    rgb + (1 - alpha[..., None]) * background,
                    alpha[..., None],
                    expected[..., None],
                    F.normalize(nnum, dim=-1),
                    l1[..., None],
                    l2[..., None],
                    median[..., None],
                    variance[..., None],
                    ambiguity[..., None],
                ) + ((nnum,) if self.mode == "gof" else ()),
                -1,
            )

        # Four original batches per dependency group. No join is placed on the
        # current stream between branches; joins bound queued working storage.
        current_stream=torch.cuda.current_stream() if self.batch_streams else None
        streams=[torch.cuda.Stream() for _ in range(4)] if self.batch_streams else []
        stream_batch_index=0
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
                if self.cuda_binning:
                    chosen = chosen_tiles[chosen_offset : chosen_offset + len(batch)]
                    chosen_offset += len(batch)
                    positions = (
                        offsets[chosen, None]
                        + torch.arange(maximum, device=means.device)[None]
                    )
                    valid = positions < offsets[chosen + 1, None]
                    if flat_ids.numel():
                        ids = torch.where(
                            valid,
                            flat_ids[positions.clamp_max(flat_ids.numel() - 1)],
                            0,
                        )
                    else:
                        ids = torch.zeros_like(positions)
                elif self.packed_membership:
                    count = len(batch)*maximum
                    ids = packed_ids[packed_offset:packed_offset+count].view(len(batch), maximum)
                    valid = packed_valid[packed_offset:packed_offset+count].view(len(batch), maximum)
                    packed_offset += count
                else:
                    indices = np.zeros((len(batch), maximum), np.int64)
                    gates = np.zeros_like(indices, dtype=bool)
                ray_rows = []
                for row, (tile_index, members_ids) in enumerate(batch):
                    if not self.cuda_binning and not self.packed_membership:
                        (
                            indices[row, : len(members_ids)],
                            gates[row, : len(members_ids)],
                        ) = members_ids, True
                    top, left, _, _ = layout[tile_index]
                    ray_rows.append(
                        rays[top : top + height, left : left + width].reshape(-1, 3)
                    )
                if not self.cuda_binning and not self.packed_membership:
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
                stream=streams[stream_batch_index%4] if streams else None
                if stream is not None:
                    stream.wait_stream(current_stream)
                    for value in (*operands,selection,background):
                        if isinstance(value,torch.Tensor) and value.is_cuda:value.record_stream(stream)
                with torch.cuda.stream(stream) if stream is not None else nullcontext():
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
                        # Custom event kernels still save per-event tensors across
                        # batches. Retain bounded-memory checkpointing for this
                        # eager CUDA path too; recomputing an individual event in
                        # its VJP does not free the whole forward autograd graph.
                        if torch.is_grad_enabled() and not self.compiled and self.native_primal_pullback is None
                        else render_batch(*operands)
                    )
                    if stream is not None:output.record_stream(current_stream)
                    for row, (tile_index, _) in enumerate(batch):
                        tiles[tile_index] = output[row].reshape(height, width, -1)
                stream_batch_index+=1
                if streams and stream_batch_index%4==0:
                    for lane in streams:current_stream.wait_stream(lane)
        if streams:
            for lane in streams:current_stream.wait_stream(lane)
        rows = []
        columns = (camera.width + tile_size - 1) // tile_size
        for start in range(0, len(tiles), columns):
            rows.append(torch.cat(tiles[start : start + columns], 1))
        combined = torch.cat(rows, 0)
        widths = (3, 1, 1, 3, 1, 1, 1, 1, 1) + ((3,) if self.mode == "gof" else ())
        channels = torch.split(combined, widths, dim=-1)
        return RenderOutput(
            *channels[:9],
            projected_means=means2d,
            projected_radii=radii,
            normal_numerator=channels[9] if self.mode == "gof" else None,
            distortion_profile=distortions,
        )
