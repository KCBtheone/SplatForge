"""Paper-derived plane homographies, reprojection gates and patch matching."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .losses import masked_mean


def sample_image(image, pixels):
    h, w = image.shape[:2]
    grid = torch.stack((2 * pixels[..., 0] / w - 1, 2 * pixels[..., 1] / h - 1), -1)
    values = F.grid_sample(
        image.permute(2, 0, 1)[None],
        grid.reshape(1, -1, 1, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return values[0, :, :, 0].T.reshape(*pixels.shape[:-1], image.shape[-1])


def multiview_losses(pipeline, sample, output, step):
    config = pipeline.config.loss
    if (
        config.multiview <= 0
        or step < config.geometric_start
        or len(pipeline.data.train_indices) < 2
    ):
        return {}
    camera = sample["camera"]
    candidates = [i for i in pipeline.data.train_indices if i != sample["index"]]
    with torch.no_grad():
        origins = torch.stack(
            [
                pipeline.data.frames[i].camera.origin.to(camera.origin)
                for i in candidates
            ]
        )
        chosen = candidates[
            int(torch.linalg.vector_norm(origins - camera.origin, dim=-1).argmin())
        ]
    neighbor = pipeline.data.sample(chosen, pipeline.model.device)
    other = neighbor["camera"]
    neighbor_output = pipeline.render(other)
    stride = config.patch_stride
    ys = (
        torch.arange(
            0, camera.height, stride, device=camera.K.device, dtype=camera.K.dtype
        )
        + 0.5
    )
    xs = (
        torch.arange(
            0, camera.width, stride, device=camera.K.device, dtype=camera.K.dtype
        )
        + 0.5
    )
    y, x = torch.meshgrid(ys, xs, indexing="ij")
    pixels = torch.stack((x, y), -1).reshape(-1, 2)
    ray = (
        torch.cat((pixels, torch.ones_like(pixels[:, :1])), 1)
        @ torch.linalg.inv(camera.K).T
    )
    depth = sample_image(output.depth, pixels).squeeze(-1)
    local = ray * depth[:, None]
    world = local @ camera.camera_to_world[:3, :3].T + camera.origin
    target = world @ other.world_to_camera[:3, :3].T + other.world_to_camera[:3, 3]
    target_h = target @ other.K.T
    uv = target_h[:, :2] / target_h[:, 2:3].clamp_min(1e-8)
    observed_depth = sample_image(neighbor_output.depth, uv).squeeze(-1)
    target_ray = (
        torch.cat((uv, torch.ones_like(uv[:, :1])), 1) @ torch.linalg.inv(other.K).T
    )
    back_world = (target_ray * observed_depth[:, None]) @ other.camera_to_world[
        :3, :3
    ].T + other.origin
    back_local = (
        back_world @ camera.world_to_camera[:3, :3].T + camera.world_to_camera[:3, 3]
    )
    back_h = back_local @ camera.K.T
    back_uv = back_h[:, :2] / back_h[:, 2:3].clamp_min(1e-8)
    reprojection = torch.linalg.vector_norm(back_uv - pixels, dim=-1)
    valid = (
        (depth > 0)
        & (observed_depth > 0)
        & (target[:, 2] > 0)
        & (uv[:, 0] >= 0.5)
        & (uv[:, 0] < other.width - 0.5)
        & (uv[:, 1] >= 0.5)
        & (uv[:, 1] < other.height - 0.5)
        & (reprojection.detach() < config.multiview_pixel_threshold)
    )
    geometry = masked_mean(
        (observed_depth - target[:, 2]).abs() / pipeline.scene_scale, valid
    )
    normals = sample_image(output.normals, pixels)
    distance = (normals * local).sum(-1)
    relative = other.world_to_camera @ camera.camera_to_world
    safe = torch.where(distance.abs() > 1e-8, distance, torch.ones_like(distance))
    plane = (
        relative[:3, :3][None]
        + relative[:3, 3][None, :, None] * normals[:, None, :] / safe[:, None, None]
    )
    homography = other.K[None] @ plane @ torch.linalg.inv(camera.K)[None]
    radius = config.patch_radius
    offsets = torch.cartesian_prod(
        torch.arange(-radius, radius + 1, device=pixels.device, dtype=pixels.dtype),
        torch.arange(-radius, radius + 1, device=pixels.device, dtype=pixels.dtype),
    )
    source_patches = pixels[:, None, :] + offsets[None]
    source_h = torch.cat((source_patches, torch.ones_like(source_patches[..., :1])), -1)
    warped = torch.einsum("nij,npj->npi", homography, source_h)
    destination = warped[..., :2] / warped[..., 2:3].clamp_min(1e-8)
    patch_valid = (
        (source_patches[..., 0] >= 0.5)
        & (source_patches[..., 0] < camera.width - 0.5)
        & (source_patches[..., 1] >= 0.5)
        & (source_patches[..., 1] < camera.height - 0.5)
        & (destination[..., 0] >= 0.5)
        & (destination[..., 0] < other.width - 0.5)
        & (destination[..., 1] >= 0.5)
        & (destination[..., 1] < other.height - 0.5)
        & valid[:, None]
    )
    a = sample_image(sample["image"], source_patches).mean(-1)
    b = sample_image(neighbor["image"], destination).mean(-1)
    correlation, patches = pipeline.ops(
        "matching.patch_zncc@1", a, b, patch_valid.to(a.dtype)
    )
    return {
        "multiview_geometry": config.multiview * config.multiview_geometry * geometry,
        "multiview_photometric": config.multiview
        * config.multiview_photometric
        * masked_mean(1 - correlation, patches),
    }
