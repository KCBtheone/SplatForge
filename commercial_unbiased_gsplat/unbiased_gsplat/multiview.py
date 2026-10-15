"""Source-isolated PGSR-style multi-view geometric and patch constraints.

Only the public PGSR paper equations 7--12 and its supplementary camera-graph
description are used here.  This module does not import, execute, or translate
the non-commercial PGSR implementation.  Pixel coordinates denote centers:
the first image sample is ``(0.5, 0.5)`` and ``grid_sample`` therefore uses
``align_corners=False``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from .models import MultiViewConsistencyResult


@dataclass(frozen=True)
class PGSRMultiViewConfig:
    """Frozen public-paper constants plus explicit numerical choices."""

    geometry_weight: float = 0.03
    photometric_weight: float = 0.15
    max_neighbors: int = 8
    max_relative_angle_degrees: float = 30.0
    min_relative_position: float = 0.01
    max_relative_position: float = 1.5
    reprojection_threshold_pixels: float = 1.0
    patch_size: int = 7
    pixel_stride: int = 4
    homography_epsilon: float = 1e-8
    variance_epsilon: float = 1e-6
    sampled_mask_threshold: float = 0.999


def validate_multiview_config(config: PGSRMultiViewConfig) -> None:
    scalar_values = {
        "geometry_weight": config.geometry_weight,
        "photometric_weight": config.photometric_weight,
        "max_relative_angle_degrees": config.max_relative_angle_degrees,
        "min_relative_position": config.min_relative_position,
        "max_relative_position": config.max_relative_position,
        "reprojection_threshold_pixels": config.reprojection_threshold_pixels,
        "homography_epsilon": config.homography_epsilon,
        "variance_epsilon": config.variance_epsilon,
        "sampled_mask_threshold": config.sampled_mask_threshold,
    }
    for name, value in scalar_values.items():
        if not torch.isfinite(torch.tensor(float(value))):
            raise ValueError(f"{name} must be finite")
    if config.geometry_weight < 0.0 or config.photometric_weight < 0.0:
        raise ValueError("multi-view loss weights must be nonnegative")
    if config.max_neighbors <= 0:
        raise ValueError("max_neighbors must be positive")
    if not 0.0 < config.max_relative_angle_degrees <= 180.0:
        raise ValueError("max_relative_angle_degrees must be in (0, 180]")
    if config.min_relative_position < 0.0:
        raise ValueError("min_relative_position must be nonnegative")
    if config.max_relative_position < config.min_relative_position:
        raise ValueError("maximum relative position must not be below minimum")
    if config.reprojection_threshold_pixels <= 0.0:
        raise ValueError("reprojection threshold must be positive")
    if config.patch_size <= 0 or config.patch_size % 2 != 1:
        raise ValueError("patch_size must be a positive odd integer")
    if config.pixel_stride <= 0:
        raise ValueError("pixel_stride must be positive")
    if config.homography_epsilon <= 0.0 or config.variance_epsilon <= 0.0:
        raise ValueError("numerical epsilons are invalid")
    if not 0.0 <= config.sampled_mask_threshold <= 1.0:
        raise ValueError("sampled_mask_threshold must be in [0, 1]")


def build_camera_neighbor_graph(
    camera_to_world: Tensor,
    source_indices: Sequence[int],
    *,
    config: PGSRMultiViewConfig = PGSRMultiViewConfig(),
) -> tuple[tuple[int, ...], ...]:
    """Build the paper's lexicographically angle-then-position image graph.

    Returned entries are dataset-item indices, not source-image indices.  The
    latter are used only as a stable final tie breaker.
    """

    validate_multiview_config(config)
    if camera_to_world.ndim != 3 or camera_to_world.shape[1:] != (4, 4):
        raise ValueError("camera_to_world must have shape [C, 4, 4]")
    if not camera_to_world.dtype.is_floating_point:
        raise TypeError("camera_to_world must have floating dtype")
    if len(source_indices) != camera_to_world.shape[0]:
        raise ValueError("source_indices must match the camera count")
    if len(set(int(index) for index in source_indices)) != len(source_indices):
        raise ValueError("source_indices must be unique")
    if not bool(torch.isfinite(camera_to_world).all().item()):
        raise ValueError("camera_to_world must be finite")

    positions = camera_to_world[:, :3, 3]
    forward = F.normalize(camera_to_world[:, :3, 2], dim=-1)
    cosine = (forward @ forward.T).clamp(-1.0, 1.0)
    angle = torch.rad2deg(torch.acos(cosine))
    distance = torch.cdist(positions, positions)

    graph: list[tuple[int, ...]] = []
    for reference in range(camera_to_world.shape[0]):
        candidates: list[tuple[float, float, int, int]] = []
        for neighbor in range(camera_to_world.shape[0]):
            if reference == neighbor:
                continue
            angle_value = float(angle[reference, neighbor].item())
            distance_value = float(distance[reference, neighbor].item())
            if angle_value > config.max_relative_angle_degrees:
                continue
            if not (
                config.min_relative_position
                <= distance_value
                <= config.max_relative_position
            ):
                continue
            candidates.append(
                (
                    angle_value,
                    distance_value,
                    int(source_indices[neighbor]),
                    neighbor,
                )
            )
        candidates.sort()
        graph.append(tuple(item[-1] for item in candidates[: config.max_neighbors]))
    return tuple(graph)


def plane_induced_homography(
    source_camera_to_world: Tensor,
    target_camera_to_world: Tensor,
    source_intrinsics: Tensor,
    target_intrinsics: Tensor,
    source_points_world: Tensor,
    source_normals_world: Tensor,
    *,
    epsilon: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    """Return ``K_t (R - t n^T / d) K_s^-1`` and its validity mask.

    The source-camera plane convention is ``n^T X + d = 0``.  Supplying a
    point on each plane therefore fixes ``d = -n^T X`` and removes the normal
    sign ambiguity because flipping both ``n`` and ``d`` leaves ``H`` intact.
    """

    if source_camera_to_world.shape != (4, 4) or target_camera_to_world.shape != (4, 4):
        raise ValueError("camera transforms must have shape [4, 4]")
    if source_intrinsics.shape != (3, 3) or target_intrinsics.shape != (3, 3):
        raise ValueError("intrinsics must have shape [3, 3]")
    if source_points_world.ndim != 2 or source_points_world.shape[-1] != 3:
        raise ValueError("source_points_world must have shape [N, 3]")
    if source_normals_world.shape != source_points_world.shape:
        raise ValueError("source normals and points must share shape")
    tensors = (
        source_camera_to_world,
        target_camera_to_world,
        source_intrinsics,
        target_intrinsics,
        source_points_world,
        source_normals_world,
    )
    if any(not tensor.dtype.is_floating_point for tensor in tensors):
        raise TypeError("homography inputs must have floating dtype")
    if len({tensor.dtype for tensor in tensors}) != 1:
        raise TypeError("homography inputs must share dtype")
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("homography inputs must share device")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")

    source_world_to_camera = torch.linalg.inv(source_camera_to_world)
    target_from_source = torch.linalg.inv(target_camera_to_world) @ source_camera_to_world
    relative_rotation = target_from_source[:3, :3]
    relative_translation = target_from_source[:3, 3]

    source_rotation = source_world_to_camera[:3, :3]
    source_translation = source_world_to_camera[:3, 3]
    points_source = source_points_world @ source_rotation.T + source_translation
    normals_source = source_normals_world @ source_rotation.T
    normal_magnitude = torch.linalg.vector_norm(normals_source, dim=-1)
    normal_ok = torch.isfinite(normals_source).all(dim=-1) & (normal_magnitude > epsilon)
    safe_magnitude = torch.where(normal_ok, normal_magnitude, torch.ones_like(normal_magnitude))
    normals_source = normals_source / safe_magnitude[:, None]
    plane_offset = -torch.sum(normals_source * points_source, dim=-1)
    valid = (
        normal_ok
        & torch.isfinite(points_source).all(dim=-1)
        & torch.isfinite(plane_offset)
        & (plane_offset.abs() > epsilon)
        & (points_source[:, 2] > epsilon)
    )
    safe_offset = torch.where(valid, plane_offset, torch.ones_like(plane_offset))
    middle = relative_rotation[None] - (
        relative_translation[None, :, None]
        * normals_source[:, None, :]
        / safe_offset[:, None, None]
    )
    homography = target_intrinsics[None] @ middle @ torch.linalg.inv(source_intrinsics)[None]
    # Torch 2.1 (the pinned AutoDL runtime) does not accept a tuple in
    # ``Tensor.all(dim=...)``.  Flatten only the two matrix axes so this keeps
    # the exact [N] validity contract across the pinned and newer runtimes.
    finite = torch.isfinite(homography).flatten(start_dim=-2).all(dim=-1)
    valid = valid & finite
    identity = torch.eye(3, dtype=homography.dtype, device=homography.device)
    homography = torch.where(valid[:, None, None], homography, identity[None])
    return homography, valid


def apply_homography(
    homography: Tensor, pixels: Tensor, *, epsilon: float = 1e-8
) -> tuple[Tensor, Tensor]:
    """Apply batched homographies to ``[N,2]`` or ``[N,P,2]`` centers."""

    if homography.ndim != 3 or homography.shape[1:] != (3, 3):
        raise ValueError("homography must have shape [N, 3, 3]")
    if pixels.ndim not in (2, 3) or pixels.shape[0] != homography.shape[0] or pixels.shape[-1] != 2:
        raise ValueError("pixels must have shape [N, 2] or [N, P, 2]")
    if homography.dtype != pixels.dtype or homography.device != pixels.device:
        raise ValueError("homography and pixels must share dtype and device")
    homogeneous = torch.cat((pixels, torch.ones_like(pixels[..., :1])), dim=-1)
    if pixels.ndim == 2:
        mapped = torch.einsum("nij,nj->ni", homography, homogeneous)
    else:
        mapped = torch.einsum("nij,npj->npi", homography, homogeneous)
    denominator = mapped[..., 2]
    valid = torch.isfinite(mapped).all(dim=-1) & (denominator.abs() > epsilon)
    safe_denominator = torch.where(valid, denominator, torch.ones_like(denominator))
    coordinates = mapped[..., :2] / safe_denominator[..., None]
    coordinates = torch.where(valid[..., None], coordinates, torch.zeros_like(coordinates))
    return coordinates, valid


def normalized_cross_correlation(
    reference_patches: Tensor,
    neighbor_patches: Tensor,
    *,
    variance_epsilon: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Return zero-mean NCC and an explicit non-degenerate-patch mask."""

    if reference_patches.shape != neighbor_patches.shape or reference_patches.ndim != 2:
        raise ValueError("patch tensors must share shape [N, P]")
    if reference_patches.dtype != neighbor_patches.dtype:
        raise TypeError("patch tensors must share dtype")
    if reference_patches.device != neighbor_patches.device:
        raise ValueError("patch tensors must share device")
    if not reference_patches.dtype.is_floating_point:
        raise TypeError("patch tensors must have floating dtype")
    if variance_epsilon <= 0.0:
        raise ValueError("variance_epsilon must be positive")
    if reference_patches.is_cuda:
        from ._cuda import cuda_available, zncc_cuda

        if cuda_available():
            return zncc_cuda(reference_patches, neighbor_patches, variance_epsilon)
    reference_centered = reference_patches - reference_patches.mean(dim=-1, keepdim=True)
    neighbor_centered = neighbor_patches - neighbor_patches.mean(dim=-1, keepdim=True)
    reference_energy = reference_centered.square().sum(dim=-1)
    neighbor_energy = neighbor_centered.square().sum(dim=-1)
    valid = (
        torch.isfinite(reference_patches).all(dim=-1)
        & torch.isfinite(neighbor_patches).all(dim=-1)
        & (reference_energy > variance_epsilon)
        & (neighbor_energy > variance_epsilon)
    )
    denominator = torch.sqrt(
        reference_energy.clamp_min(variance_epsilon)
        * neighbor_energy.clamp_min(variance_epsilon)
    )
    raw = (reference_centered * neighbor_centered).sum(dim=-1) / denominator
    correlation = torch.where(valid, raw.clamp(-1.0, 1.0), torch.zeros_like(raw))
    return correlation, valid


def normalize_rendered_plane_normals(
    rendered_normals_world: Tensor,
    rendered_alphas: Tensor,
    corrected_depth_valid: Tensor,
    *,
    epsilon: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    """Normalize gsplat's accumulated world normal into a per-pixel plane."""

    if rendered_normals_world.ndim == 4 and rendered_normals_world.shape[0] == 1:
        rendered_normals_world = rendered_normals_world[0]
    if rendered_alphas.ndim == 4 and rendered_alphas.shape[0] == 1:
        rendered_alphas = rendered_alphas[0]
    if rendered_alphas.shape[-1:] == (1,):
        rendered_alphas = rendered_alphas[..., 0]
    if corrected_depth_valid.ndim == 3 and corrected_depth_valid.shape[0] == 1:
        corrected_depth_valid = corrected_depth_valid[0]
    if rendered_normals_world.ndim != 3 or rendered_normals_world.shape[-1] != 3:
        raise ValueError("rendered normals must have shape [H, W, 3]")
    if rendered_alphas.shape != rendered_normals_world.shape[:2]:
        raise ValueError("rendered alpha shape does not match normals")
    if corrected_depth_valid.shape != rendered_alphas.shape or corrected_depth_valid.dtype != torch.bool:
        raise ValueError("corrected_depth_valid must be boolean [H, W]")
    magnitude = torch.linalg.vector_norm(rendered_normals_world, dim=-1)
    valid = (
        corrected_depth_valid
        & torch.isfinite(rendered_normals_world).all(dim=-1)
        & torch.isfinite(rendered_alphas)
        & (rendered_alphas > 0.0)
        & (magnitude > epsilon)
    )
    safe = torch.where(valid, magnitude, torch.ones_like(magnitude))
    normals = rendered_normals_world / safe[..., None]
    normals = torch.where(valid[..., None], normals, torch.zeros_like(normals))
    return normals, valid


def _in_bounds(pixels: Tensor, *, width: int, height: int) -> Tensor:
    return (
        torch.isfinite(pixels).all(dim=-1)
        & (pixels[..., 0] >= 0.5)
        & (pixels[..., 0] <= width - 0.5)
        & (pixels[..., 1] >= 0.5)
        & (pixels[..., 1] <= height - 0.5)
    )


def _sample_hwc(image: Tensor, pixels: Tensor) -> Tensor:
    if image.ndim != 3:
        raise ValueError("sampled image must have shape [H, W, C]")
    height, width = image.shape[:2]
    original_ndim = pixels.ndim
    if original_ndim == 2:
        pixels = pixels[:, None, :]
    if pixels.ndim != 3 or pixels.shape[-1] != 2:
        raise ValueError("pixels must have shape [N, 2] or [N, P, 2]")
    normalized = torch.stack(
        (
            2.0 * pixels[..., 0] / width - 1.0,
            2.0 * pixels[..., 1] / height - 1.0,
        ),
        dim=-1,
    )
    sampled = F.grid_sample(
        image.permute(2, 0, 1)[None],
        normalized[None],
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )[0].permute(1, 2, 0)
    return sampled[:, 0] if original_ndim == 2 else sampled


def _grayscale(image: Tensor) -> Tensor:
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("images must have shape [H, W, 3]")
    weights = image.new_tensor((0.299, 0.587, 0.114))
    return torch.sum(image * weights, dim=-1, keepdim=True)


def _unfold_fixed_patches(
    image: Tensor, *, patch_size: int, pixel_stride: int, expected_rows: int
) -> Tensor:
    """Extract stride-spaced ``[L, P]`` patches without a sampling grid.

    Centers start at ``radius`` with step ``pixel_stride`` while staying
    strictly inside the ``radius`` border, exactly matching the centers used
    by the grid sampler. Bilinear sampling at exact pixel centers reproduces
    pixel values, so ``unfold`` is exact here; it additionally keeps the
    ground-truth reference side out of the autograd graph and avoids
    materializing the ``[N, P, 2]`` reference grid.
    """

    if image.ndim != 3 or image.shape[-1] != 1:
        raise ValueError("fixed patches are extracted from [H, W, 1] images")
    if patch_size <= 0 or patch_size % 2 != 1:
        raise ValueError("patch_size must be a positive odd integer")
    if pixel_stride <= 0:
        raise ValueError("pixel_stride must be positive")
    patches = F.unfold(
        image.permute(2, 0, 1)[None],
        kernel_size=patch_size,
        stride=pixel_stride,
    )[0].transpose(0, 1)
    if patches.shape[0] != expected_rows or patches.shape[1] != patch_size * patch_size:
        raise ValueError(
            "unfold patch layout mismatch: got "
            f"{tuple(patches.shape)}, expected "
            f"({expected_rows}, {patch_size * patch_size})"
        )
    return patches


def _zero_result(anchor: Tensor) -> MultiViewConsistencyResult:
    zero = anchor.sum() * 0.0
    empty_scalar = anchor.new_empty((0,))
    empty_bool = torch.empty((0,), dtype=torch.bool, device=anchor.device)
    empty_pixels = anchor.new_empty((0, 2))
    return MultiViewConsistencyResult(
        geometry_loss=zero,
        photometric_loss=zero,
        geometry_numerator=zero,
        geometry_denominator=zero.detach(),
        photometric_numerator=zero,
        photometric_denominator=zero.detach(),
        reprojection_error=empty_scalar,
        occlusion_weight=empty_scalar,
        geometry_valid_mask=empty_bool,
        photometric_valid_mask=empty_bool,
        reference_pixels=empty_pixels,
        neighbor_pixels=empty_pixels,
    )


def multiview_consistency(
    reference_image: Tensor,
    neighbor_image: Tensor,
    reference_points_world: Tensor,
    reference_normals_world: Tensor,
    reference_surface_valid: Tensor,
    neighbor_points_world: Tensor,
    neighbor_normals_world: Tensor,
    neighbor_surface_valid: Tensor,
    reference_mask: Tensor,
    neighbor_mask: Tensor,
    reference_camera_to_world: Tensor,
    neighbor_camera_to_world: Tensor,
    reference_intrinsics: Tensor,
    neighbor_intrinsics: Tensor,
    *,
    config: PGSRMultiViewConfig = PGSRMultiViewConfig(),
) -> MultiViewConsistencyResult:
    """Evaluate PGSR-style forward/backward and 7x7 patch consistency."""

    validate_multiview_config(config)
    if reference_image.shape != neighbor_image.shape or reference_image.ndim != 3:
        raise ValueError("reference and neighbor images must share shape [H, W, 3]")
    height, width, channels = reference_image.shape
    if channels != 3:
        raise ValueError("multi-view images must have three channels")
    expected_vectors = (height, width, 3)
    expected_masks = (height, width)
    for name, tensor in (
        ("reference_points_world", reference_points_world),
        ("reference_normals_world", reference_normals_world),
        ("neighbor_points_world", neighbor_points_world),
        ("neighbor_normals_world", neighbor_normals_world),
    ):
        if tensor.shape != expected_vectors:
            raise ValueError(f"{name} must have shape [H, W, 3]")
    for name, tensor in (
        ("reference_surface_valid", reference_surface_valid),
        ("neighbor_surface_valid", neighbor_surface_valid),
        ("reference_mask", reference_mask),
        ("neighbor_mask", neighbor_mask),
    ):
        if tensor.shape != expected_masks or tensor.dtype != torch.bool:
            raise ValueError(f"{name} must be boolean [H, W]")
    floating = (
        reference_image,
        neighbor_image,
        reference_points_world,
        reference_normals_world,
        neighbor_points_world,
        neighbor_normals_world,
        reference_camera_to_world,
        neighbor_camera_to_world,
        reference_intrinsics,
        neighbor_intrinsics,
    )
    if any(not tensor.dtype.is_floating_point for tensor in floating):
        raise TypeError("multi-view floating inputs must have floating dtype")
    if len({tensor.dtype for tensor in floating}) != 1:
        raise TypeError("multi-view floating inputs must share dtype")
    all_inputs = floating + (
        reference_surface_valid,
        neighbor_surface_valid,
        reference_mask,
        neighbor_mask,
    )
    if len({tensor.device for tensor in all_inputs}) != 1:
        raise ValueError("all multi-view inputs must share device")

    radius = config.patch_size // 2
    y_indices = torch.arange(
        radius, height - radius, config.pixel_stride, device=reference_image.device
    )
    x_indices = torch.arange(
        radius, width - radius, config.pixel_stride, device=reference_image.device
    )
    if y_indices.numel() == 0 or x_indices.numel() == 0:
        return _zero_result(reference_points_world)
    y_grid, x_grid = torch.meshgrid(y_indices, x_indices, indexing="ij")
    y_flat = y_grid.reshape(-1)
    x_flat = x_grid.reshape(-1)
    reference_pixels = torch.stack(
        (x_flat.to(reference_image.dtype) + 0.5, y_flat.to(reference_image.dtype) + 0.5),
        dim=-1,
    )
    reference_points = reference_points_world[y_flat, x_flat]
    reference_normals = reference_normals_world[y_flat, x_flat]
    reference_center_valid = (
        reference_surface_valid[y_flat, x_flat] & reference_mask[y_flat, x_flat]
    )

    forward_h, forward_plane_valid = plane_induced_homography(
        reference_camera_to_world,
        neighbor_camera_to_world,
        reference_intrinsics,
        neighbor_intrinsics,
        reference_points,
        reference_normals,
        epsilon=config.homography_epsilon,
    )
    neighbor_pixels, forward_projection_valid = apply_homography(
        forward_h, reference_pixels, epsilon=config.homography_epsilon
    )
    neighbor_center_in_bounds = _in_bounds(neighbor_pixels, width=width, height=height)
    # One fused sample for points/normals/validity/mask: identical per-element
    # bilinear values with a single graph node instead of four.
    neighbor_geo_buffer = torch.cat(
        (
            neighbor_points_world,
            neighbor_normals_world,
            neighbor_surface_valid.to(reference_image.dtype)[..., None],
            neighbor_mask.to(reference_image.dtype)[..., None],
        ),
        dim=-1,
    )
    sampled_geo = _sample_hwc(neighbor_geo_buffer, neighbor_pixels)
    sampled_neighbor_points = sampled_geo[..., :3]
    sampled_neighbor_normals = sampled_geo[..., 3:6]
    sampled_neighbor_valid = sampled_geo[..., 6] >= config.sampled_mask_threshold
    sampled_neighbor_mask = sampled_geo[..., 7] >= config.sampled_mask_threshold

    reverse_h, reverse_plane_valid = plane_induced_homography(
        neighbor_camera_to_world,
        reference_camera_to_world,
        neighbor_intrinsics,
        reference_intrinsics,
        sampled_neighbor_points,
        sampled_neighbor_normals,
        epsilon=config.homography_epsilon,
    )
    backward_pixels, backward_projection_valid = apply_homography(
        reverse_h, neighbor_pixels, epsilon=config.homography_epsilon
    )
    reprojection_error = torch.linalg.vector_norm(
        backward_pixels - reference_pixels, dim=-1
    )
    base_valid = (
        reference_center_valid
        & forward_plane_valid
        & forward_projection_valid
        & neighbor_center_in_bounds
        & sampled_neighbor_valid
        & sampled_neighbor_mask
        & reverse_plane_valid
        & backward_projection_valid
        & torch.isfinite(reprojection_error)
    )
    geometry_valid = base_valid & (
        reprojection_error < config.reprojection_threshold_pixels
    )
    occlusion_weight = torch.where(
        geometry_valid,
        torch.exp(-reprojection_error).detach(),
        torch.zeros_like(reprojection_error),
    )
    geometry_terms = torch.where(
        geometry_valid,
        occlusion_weight * reprojection_error,
        torch.zeros_like(reprojection_error),
    )
    geometry_numerator = geometry_terms.sum()
    geometry_denominator = geometry_valid.sum().to(reference_image.dtype)
    geometry_loss = geometry_numerator / geometry_denominator.clamp_min(1.0)

    offsets_y, offsets_x = torch.meshgrid(
        torch.arange(-radius, radius + 1, device=reference_image.device),
        torch.arange(-radius, radius + 1, device=reference_image.device),
        indexing="ij",
    )
    offsets = torch.stack((offsets_x, offsets_y), dim=-1).reshape(-1, 2)
    offsets = offsets.to(reference_image.dtype)
    reference_patch_pixels = reference_pixels[:, None, :] + offsets[None]
    neighbor_patch_pixels, patch_projection_valid = apply_homography(
        forward_h, reference_patch_pixels, epsilon=config.homography_epsilon
    )
    neighbor_patch_in_bounds = _in_bounds(
        neighbor_patch_pixels, width=width, height=height
    ).all(dim=-1)

    reference_gray = _grayscale(reference_image)
    neighbor_gray = _grayscale(neighbor_image)
    reference_patches = _unfold_fixed_patches(
        reference_gray,
        patch_size=config.patch_size,
        pixel_stride=config.pixel_stride,
        expected_rows=int(reference_pixels.shape[0]),
    )
    reference_patch_mask = _unfold_fixed_patches(
        reference_mask.to(reference_image.dtype)[..., None],
        patch_size=config.patch_size,
        pixel_stride=config.pixel_stride,
        expected_rows=int(reference_pixels.shape[0]),
    )
    # One fused sample for neighbor gray + mask at the warped patch grid.
    neighbor_patch_source = torch.cat(
        (
            neighbor_gray,
            neighbor_mask.to(reference_image.dtype)[..., None],
        ),
        dim=-1,
    )
    sampled_neighbor_patch = _sample_hwc(neighbor_patch_source, neighbor_patch_pixels)
    neighbor_patches = sampled_neighbor_patch[..., 0]
    neighbor_patch_mask = sampled_neighbor_patch[..., 1]
    ncc, variance_valid = normalized_cross_correlation(
        reference_patches,
        neighbor_patches,
        variance_epsilon=config.variance_epsilon,
    )
    photometric_valid = (
        geometry_valid
        & patch_projection_valid.all(dim=-1)
        & neighbor_patch_in_bounds
        & (reference_patch_mask >= config.sampled_mask_threshold).all(dim=-1)
        & (neighbor_patch_mask >= config.sampled_mask_threshold).all(dim=-1)
        & variance_valid
    )
    photometric_terms = torch.where(
        photometric_valid,
        occlusion_weight * (1.0 - ncc),
        torch.zeros_like(ncc),
    )
    photometric_numerator = photometric_terms.sum()
    photometric_denominator = photometric_valid.sum().to(reference_image.dtype)
    photometric_loss = photometric_numerator / photometric_denominator.clamp_min(1.0)

    return MultiViewConsistencyResult(
        geometry_loss=geometry_loss,
        photometric_loss=photometric_loss,
        geometry_numerator=geometry_numerator,
        geometry_denominator=geometry_denominator,
        photometric_numerator=photometric_numerator,
        photometric_denominator=photometric_denominator,
        reprojection_error=torch.where(
            base_valid, reprojection_error, torch.zeros_like(reprojection_error)
        ),
        occlusion_weight=occlusion_weight,
        geometry_valid_mask=geometry_valid,
        photometric_valid_mask=photometric_valid,
        reference_pixels=reference_pixels,
        neighbor_pixels=neighbor_pixels,
    )


def weighted_multiview_regularization(
    result: MultiViewConsistencyResult,
    config: PGSRMultiViewConfig = PGSRMultiViewConfig(),
) -> Tensor:
    """Combine the two public-paper multi-view terms with explicit weights."""

    validate_multiview_config(config)
    return (
        config.geometry_weight * result.geometry_loss
        + config.photometric_weight * result.photometric_loss
    )
