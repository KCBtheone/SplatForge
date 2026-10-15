"""Thin training adapter; all mathematical work remains in tensor-level modules."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .convergence import Reduction, depth_convergence
from .corrected_depth import corrected_depth, segmented_cumulative_sum
from .depth_normal import depth_to_normal, depth_to_points
from .hit_sequence import collect_ordered_hits, gather_hit_attributes
from .losses import (
    NormalReduction,
    normal_consistency_from_rendered,
    normal_consistency_loss,
)
from .models import (
    ConvergenceResult,
    CorrectedSurfaceResult,
    CorrectedDepthResult,
    DenseNormalConsistencyResult,
    DepthNormalResult,
    HitSequence,
    NormalConsistencyResult,
    RaySplatSamples,
)
from .ray_splat_depth import ray_splat_depth_and_weight


@dataclass(frozen=True)
class UnbiasedSurfelConfig:
    opacity_epsilon: float = 0.1
    corrected_depth_threshold: float = 0.6
    far_gradient_scale: float = 1.25
    convergence_weight: float = 7.0
    normal_weight: float = 0.05
    dssim_weight: float = 0.2
    densification_stop_iteration: int = 20_000
    dtu_gamma: float = 0.5
    convergence_reduction: Reduction = "mean_segments"
    normal_reduction: NormalReduction = "mean_segments"
    hz_epsilon: float = 0.0
    normal_epsilon: float = 1e-12


@dataclass(frozen=True)
class UnbiasedAuxiliaries:
    sequence: HitSequence
    samples: RaySplatSamples
    convergence: ConvergenceResult
    corrected_depth: CorrectedDepthResult
    depth_normal: DepthNormalResult
    normal_consistency: NormalConsistencyResult | DenseNormalConsistencyResult


def compute_corrected_surface_from_hit_sequence(
    meta: Mapping[str, Any],
    sequence: HitSequence,
    viewmats: Tensor,
    intrinsics: Tensor,
    *,
    config: UnbiasedSurfelConfig = UnbiasedSurfelConfig(),
) -> CorrectedSurfaceResult:
    """Return only the dense corrected surface needed by a neighbor view.

    Unlike :func:`compute_from_hit_sequence`, this path does not construct the
    convergence or normal-consistency losses. More importantly, none of its
    per-hit samples or cumulative corrected-opacity buffers escape the
    function. The caller can therefore evaluate a no-gradient neighbor render
    without keeping tens of millions of hit values alive during the subsequent
    dense multi-view calculation.
    """

    validate_config(config)
    gathered = gather_hit_attributes(meta, sequence)
    samples = ray_splat_depth_and_weight(
        gathered.ray_transforms,
        gathered.means2d,
        sequence.pixel_ids,
        sequence.image_width,
        hz_epsilon=config.hz_epsilon,
    )
    corrected = corrected_depth(
        samples.depth,
        gathered.opacities,
        samples.gaussian_weight,
        sequence.segment_ids,
        samples.valid_mask,
        num_segments=sequence.num_segments,
        opacity_epsilon=config.opacity_epsilon,
        threshold=config.corrected_depth_threshold,
    )
    flat_viewmats, flat_intrinsics = _flatten_camera_matrices(
        viewmats, intrinsics, sequence.num_images
    )
    depth_images = corrected.depth.reshape(
        sequence.num_images, sequence.image_height, sequence.image_width
    )
    corrected_valid = corrected.valid_mask.reshape(
        sequence.num_images, sequence.image_height, sequence.image_width
    )
    points, point_valid = depth_to_points(
        depth_images,
        corrected_valid,
        torch.linalg.inv(flat_viewmats),
        flat_intrinsics,
    )
    return CorrectedSurfaceResult(
        points=points,
        valid_mask=point_valid,
        num_hits=sequence.num_hits,
        corrected_valid_pixels=int(corrected.valid_mask.sum().item()),
    )


@torch.no_grad()
def compute_corrected_surface_chunked_from_hit_sequence(
    meta: Mapping[str, Any],
    sequence: HitSequence,
    viewmats: Tensor,
    intrinsics: Tensor,
    *,
    max_hits_per_chunk: int = 1_048_576,
    config: UnbiasedSurfelConfig = UnbiasedSurfelConfig(),
) -> CorrectedSurfaceResult:
    """Compute the exact no-gradient corrected surface in ordered hit chunks.

    Chunks never alter the hit order. A scalar carry is applied only when one
    dense pixel segment crosses a chunk boundary, so the first strict
    corrected-opacity threshold crossing is identical to the unchunked scan.
    This function is intentionally no-gradient and is only for the explicitly
    one-sided multi-view neighbor variant.
    """

    validate_config(config)
    if max_hits_per_chunk <= 0:
        raise ValueError("max_hits_per_chunk must be positive")
    required = ("means2d", "ray_transforms", "opacities")
    missing = [name for name in required if name not in meta]
    if missing:
        raise KeyError(f"gsplat metadata is missing required keys: {missing}")
    means2d = meta["means2d"]
    ray_transforms = meta["ray_transforms"]
    opacities = meta["opacities"]
    if not all(isinstance(value, Tensor) for value in (means2d, ray_transforms, opacities)):
        raise TypeError("gsplat corrected-surface metadata must be tensors")
    if means2d.ndim < 3 or means2d.shape[-1] != 2:
        raise ValueError("unexpected gsplat means2d schema")
    if ray_transforms.shape[:-2] != means2d.shape[:-1] or ray_transforms.shape[-2:] != (3, 3):
        raise ValueError("unexpected gsplat ray_transforms schema")
    if opacities.shape != means2d.shape[:-1]:
        raise ValueError("unexpected gsplat opacities schema")

    num_gaussians = int(means2d.shape[-2])
    flat_means2d = means2d.reshape(-1, num_gaussians, 2)
    flat_transforms = ray_transforms.reshape(-1, num_gaussians, 3, 3)
    flat_opacities = opacities.reshape(-1, num_gaussians)
    if int(flat_means2d.shape[0]) != sequence.num_images:
        raise ValueError("metadata image count does not match the hit sequence")

    dense_depth = means2d.new_zeros((sequence.num_segments,))
    dense_valid = torch.zeros(
        sequence.num_segments, dtype=torch.bool, device=means2d.device
    )
    carry_segment = -1
    carry_value = means2d.new_zeros(())
    for start in range(0, sequence.num_hits, max_hits_per_chunk):
        end = min(start + max_hits_per_chunk, sequence.num_hits)
        gaussian_ids = sequence.gaussian_ids[start:end]
        image_ids = sequence.image_ids[start:end]
        pixel_ids = sequence.pixel_ids[start:end]
        segment_ids = sequence.segment_ids[start:end]
        chunk_means2d = flat_means2d[image_ids, gaussian_ids]
        chunk_transforms = flat_transforms[image_ids, gaussian_ids]
        chunk_opacities = flat_opacities[image_ids, gaussian_ids]
        samples = ray_splat_depth_and_weight(
            chunk_transforms,
            chunk_means2d,
            pixel_ids,
            sequence.image_width,
            hz_epsilon=config.hz_epsilon,
        )
        raw_contribution = (
            chunk_opacities + config.opacity_epsilon
        ) * samples.gaussian_weight
        contribution = torch.where(
            samples.valid_mask & torch.isfinite(raw_contribution),
            raw_contribution,
            torch.zeros_like(raw_contribution),
        )
        cumulative = segmented_cumulative_sum(contribution, segment_ids)
        first_segment = int(segment_ids[0].item())
        if first_segment == carry_segment:
            cumulative = cumulative + (segment_ids == first_segment) * carry_value

        crossing = (
            samples.valid_mask
            & torch.isfinite(cumulative)
            & (cumulative > config.corrected_depth_threshold)
        )
        crossing_indices = torch.nonzero(crossing, as_tuple=False).flatten()
        if crossing_indices.numel():
            crossing_segments = segment_ids[crossing_indices]
            first_crossing = torch.ones_like(crossing_segments, dtype=torch.bool)
            first_crossing[1:] = crossing_segments[1:] != crossing_segments[:-1]
            selected_indices = crossing_indices[first_crossing]
            selected_segments = segment_ids[selected_indices]
            new_segments = ~dense_valid[selected_segments]
            selected_indices = selected_indices[new_segments]
            selected_segments = selected_segments[new_segments]
            dense_depth[selected_segments] = samples.depth[selected_indices]
            dense_valid[selected_segments] = True

        carry_segment = int(segment_ids[-1].item())
        carry_value = cumulative[-1]

    flat_viewmats, flat_intrinsics = _flatten_camera_matrices(
        viewmats, intrinsics, sequence.num_images
    )
    depth_images = dense_depth.reshape(
        sequence.num_images, sequence.image_height, sequence.image_width
    )
    valid_images = dense_valid.reshape(
        sequence.num_images, sequence.image_height, sequence.image_width
    )
    points, point_valid = depth_to_points(
        depth_images,
        valid_images,
        torch.linalg.inv(flat_viewmats),
        flat_intrinsics,
    )
    return CorrectedSurfaceResult(
        points=points,
        valid_mask=point_valid,
        num_hits=sequence.num_hits,
        corrected_valid_pixels=int(dense_valid.sum().item()),
    )


def validate_config(config: UnbiasedSurfelConfig) -> None:
    if config.opacity_epsilon < 0.0:
        raise ValueError("opacity_epsilon must be nonnegative")
    if config.corrected_depth_threshold < 0.0:
        raise ValueError("corrected_depth_threshold must be nonnegative")
    if config.far_gradient_scale <= 0.0:
        raise ValueError("far_gradient_scale must be positive")
    if config.convergence_weight < 0.0 or config.normal_weight < 0.0:
        raise ValueError("loss weights must be nonnegative")
    if not 0.0 <= config.dssim_weight <= 1.0:
        raise ValueError("dssim_weight must be in [0, 1]")
    if config.densification_stop_iteration < 0:
        raise ValueError("densification_stop_iteration must be nonnegative")
    if config.dtu_gamma <= 0.0:
        raise ValueError("dtu_gamma must be positive")


def _flatten_camera_matrices(
    viewmats: Tensor, intrinsics: Tensor, num_images: int
) -> tuple[Tensor, Tensor]:
    if viewmats.shape[-2:] != (4, 4) or intrinsics.shape[-2:] != (3, 3):
        raise ValueError("viewmats/intrinsics must end in [4,4]/[3,3]")
    flat_viewmats = viewmats.reshape(-1, 4, 4)
    flat_intrinsics = intrinsics.reshape(-1, 3, 3)
    if flat_viewmats.shape[0] != num_images or flat_intrinsics.shape[0] != num_images:
        raise ValueError("camera matrix count does not match gsplat metadata")
    return flat_viewmats, flat_intrinsics


def compute_from_hit_sequence(
    meta: Mapping[str, Any],
    sequence: HitSequence,
    viewmats: Tensor,
    intrinsics: Tensor,
    *,
    scene_radius: float,
    config: UnbiasedSurfelConfig = UnbiasedSurfelConfig(),
    rendered_alphas: Tensor | None = None,
    rendered_normals: Tensor | None = None,
) -> UnbiasedAuxiliaries:
    """Compute all Unbiased Surfel auxiliaries from a frozen hit sequence."""

    validate_config(config)
    gathered = gather_hit_attributes(meta, sequence)
    if gathered.normals is None:
        raise KeyError("gsplat metadata must expose normals for normal supervision")
    samples = ray_splat_depth_and_weight(
        gathered.ray_transforms,
        gathered.means2d,
        sequence.pixel_ids,
        sequence.image_width,
        hz_epsilon=config.hz_epsilon,
    )
    convergence = depth_convergence(
        samples.depth,
        samples.gaussian_weight,
        sequence.segment_ids,
        samples.valid_mask,
        scene_radius=scene_radius,
        num_segments=sequence.num_segments,
        reduction=config.convergence_reduction,
        far_gradient_scale=config.far_gradient_scale,
    )
    corrected = corrected_depth(
        samples.depth,
        gathered.opacities,
        samples.gaussian_weight,
        sequence.segment_ids,
        samples.valid_mask,
        num_segments=sequence.num_segments,
        opacity_epsilon=config.opacity_epsilon,
        threshold=config.corrected_depth_threshold,
    )

    flat_viewmats, flat_intrinsics = _flatten_camera_matrices(
        viewmats, intrinsics, sequence.num_images
    )
    camera_to_world = torch.linalg.inv(flat_viewmats)
    depth_images = corrected.depth.reshape(
        sequence.num_images, sequence.image_height, sequence.image_width
    )
    depth_valid_images = corrected.valid_mask.reshape(
        sequence.num_images, sequence.image_height, sequence.image_width
    )
    depth_normal = depth_to_normal(
        depth_images,
        depth_valid_images,
        camera_to_world,
        flat_intrinsics,
        normal_epsilon=config.normal_epsilon,
        orient_toward_camera=True,
    )

    if (rendered_alphas is None) != (rendered_normals is None):
        raise ValueError("rendered_alphas and rendered_normals must be provided together")
    if rendered_alphas is not None and rendered_normals is not None:
        if config.normal_reduction == "mean_hits":
            raise ValueError("mean_hits is unavailable for dense primary-renderer outputs")
        normal_consistency = normal_consistency_from_rendered(
            rendered_normals.reshape(
                sequence.num_images, sequence.image_height, sequence.image_width, 3
            ),
            rendered_alphas.reshape(
                sequence.num_images, sequence.image_height, sequence.image_width, 1
            ),
            depth_normal.normals,
            depth_normal.valid_mask,
            reduction=config.normal_reduction,
            normalization_epsilon=config.normal_epsilon,
        )
    else:
        rotations = camera_to_world[:, :3, :3]
        world_surfel_normals = torch.einsum(
            "mij,mj->mi", rotations[sequence.image_ids], gathered.normals
        )
        normal_consistency = normal_consistency_loss(
            world_surfel_normals,
            depth_normal.normals.reshape(sequence.num_segments, 3),
            depth_normal.valid_mask.reshape(sequence.num_segments),
            gathered.opacities,
            samples.gaussian_weight,
            sequence.segment_ids,
            samples.valid_mask,
            num_segments=sequence.num_segments,
            reduction=config.normal_reduction,
            normalization_epsilon=config.normal_epsilon,
        )
    return UnbiasedAuxiliaries(
        sequence=sequence,
        samples=samples,
        convergence=convergence,
        corrected_depth=corrected,
        depth_normal=depth_normal,
        normal_consistency=normal_consistency,
    )


def compute_from_gsplat_meta(
    meta: Mapping[str, Any],
    viewmats: Tensor,
    intrinsics: Tensor,
    *,
    scene_radius: float,
    config: UnbiasedSurfelConfig = UnbiasedSurfelConfig(),
) -> UnbiasedAuxiliaries:
    """Collect gsplat's accepted unpacked hits and compute all auxiliaries."""

    sequence = collect_ordered_hits(meta)
    return compute_from_hit_sequence(
        meta,
        sequence,
        viewmats,
        intrinsics,
        scene_radius=scene_radius,
        config=config,
    )


def compute_corrected_surface_from_gsplat_meta(
    meta: Mapping[str, Any],
    viewmats: Tensor,
    intrinsics: Tensor,
    *,
    config: UnbiasedSurfelConfig = UnbiasedSurfelConfig(),
) -> CorrectedSurfaceResult:
    """Collect ordered hits and return a compact dense corrected surface."""

    sequence = collect_ordered_hits(meta)
    return compute_corrected_surface_from_hit_sequence(
        meta,
        sequence,
        viewmats,
        intrinsics,
        config=config,
    )


@torch.no_grad()
def compute_corrected_surface_chunked_from_gsplat_meta(
    meta: Mapping[str, Any],
    viewmats: Tensor,
    intrinsics: Tensor,
    *,
    max_hits_per_chunk: int = 1_048_576,
    config: UnbiasedSurfelConfig = UnbiasedSurfelConfig(),
) -> CorrectedSurfaceResult:
    """Neighbor surface: fused tile raster when CUDA is available, else chunked hits."""

    from ._cuda import cuda_available, rasterize_unbiased_aux

    if cuda_available() and meta["means2d"].is_cuda:
        if int(meta["isect_offsets"].shape[0]) != 1:
            raise ValueError(
                "fused neighbor raster currently supports a single image per call"
            )
        means2d = meta["means2d"].reshape(-1, 2)
        ray_transforms = meta["ray_transforms"].reshape(-1, 9)
        opacities = meta["opacities"].reshape(-1)
        depth, valid, _conv, _sel = rasterize_unbiased_aux(
            means2d,
            ray_transforms,
            opacities,
            meta["isect_offsets"].to(torch.int32),
            meta["flatten_ids"].to(torch.int32),
            int(meta["width"]),
            int(meta["height"]),
            int(meta["tile_size"]),
            config.opacity_epsilon,
            config.corrected_depth_threshold,
            1.0,
            config.far_gradient_scale,
        )
        flat_viewmats, flat_intrinsics = _flatten_camera_matrices(
            viewmats, intrinsics, int(depth.shape[0])
        )
        points, point_valid = depth_to_points(
            depth,
            valid,
            torch.linalg.inv(flat_viewmats),
            flat_intrinsics,
        )
        return CorrectedSurfaceResult(
            points=points,
            valid_mask=point_valid,
            num_hits=0,
            corrected_valid_pixels=0,
        )

    sequence = collect_ordered_hits(meta)
    return compute_corrected_surface_chunked_from_hit_sequence(
        meta,
        sequence,
        viewmats,
        intrinsics,
        max_hits_per_chunk=max_hits_per_chunk,
        config=config,
    )


def _compute_from_fused_raster(
    rendered_alphas: Tensor,
    rendered_normals: Tensor,
    meta: Mapping[str, Any],
    viewmats: Tensor,
    intrinsics: Tensor,
    *,
    scene_radius: float,
    config: UnbiasedSurfelConfig,
) -> UnbiasedAuxiliaries:
    from ._cuda import rasterize_unbiased_aux

    tile_offsets = meta["isect_offsets"].to(torch.int32)
    # The fused kernel indexes per-image Gaussian tables as image_id * N + g.
    # The flattened [I*N, ...] layout is only exact for a single image; guard
    # multi-image batches until the kernel takes an explicit num_gaussians.
    num_images_meta = int(tile_offsets.shape[0])
    if num_images_meta != 1:
        raise ValueError(
            "fused unbiased raster currently supports a single image per call; "
            f"got {num_images_meta} (use the torch hit-sequence path instead)"
        )
    if config.convergence_reduction == "mean_pairs":
        raise ValueError(
            "mean_pairs is unavailable for fused raster outputs (no pair count); "
            "use the torch hit-sequence path or mean_active_segments"
        )
    means2d = meta["means2d"].reshape(-1, 2)
    ray_transforms = meta["ray_transforms"].reshape(-1, 9)
    opacities = meta["opacities"].reshape(-1)
    flatten_ids = meta["flatten_ids"].to(torch.int32)
    width = int(meta["width"])
    height = int(meta["height"])
    tile_size = int(meta["tile_size"])
    depth, valid, conv, sel = rasterize_unbiased_aux(
        means2d,
        ray_transforms,
        opacities,
        tile_offsets,
        flatten_ids,
        width,
        height,
        tile_size,
        config.opacity_epsilon,
        config.corrected_depth_threshold,
        scene_radius,
        config.far_gradient_scale,
    )
    num_images = int(depth.shape[0])
    device = depth.device
    empty_hits = torch.zeros((0,), dtype=torch.int64, device=device)
    sequence = HitSequence(
        gaussian_ids=empty_hits,
        pixel_ids=empty_hits,
        image_ids=empty_hits,
        segment_ids=empty_hits,
        num_images=num_images,
        image_height=height,
        image_width=width,
    )
    samples = RaySplatSamples(
        gaussian_weight=depth.new_zeros((0,)),
        depth=depth.new_zeros((0,)),
        rho3=depth.new_zeros((0,)),
        rho2=depth.new_zeros((0,)),
        local_uv=depth.new_zeros((0, 2)),
        ray_plane_mask=torch.zeros((0,), dtype=torch.bool, device=device),
        valid_mask=torch.zeros((0,), dtype=torch.bool, device=device),
    )
    conv_sum = conv.sum()
    num_segments = num_images * height * width
    if config.convergence_reduction == "sum":
        denom = depth.new_tensor(1.0)
    elif config.convergence_reduction == "mean_segments":
        denom = depth.new_tensor(float(num_segments))
    elif config.convergence_reduction == "mean_active_segments":
        # Fused path has no explicit pair list; a pixel is active when its
        # accumulated pair loss is nonzero. Pixels with valid pairs whose
        # deltas are exactly zero are measure-zero and ignored here.
        denom = (conv.reshape(-1) != 0).sum().to(conv.dtype)
    else:  # mean_pairs is rejected above
        raise ValueError(
            f"unsupported fused convergence reduction: {config.convergence_reduction!r}"
        )
    convergence = ConvergenceResult(
        loss=conv_sum / denom.clamp_min(1.0),
        numerator=conv_sum,
        denominator=denom,
        per_pair=conv.reshape(-1),
        pair_mask=conv.reshape(-1) > 0,
        previous_indices=empty_hits,
        current_indices=empty_hits,
        per_segment=conv.reshape(-1),
    )
    corrected = CorrectedDepthResult(
        depth=depth.reshape(-1),
        valid_mask=valid.reshape(-1),
        selected_hit_indices=sel.reshape(-1).to(torch.int64),
        contribution=depth.new_zeros((0,)),
        cumulative_contribution=depth.new_zeros((0,)),
    )
    flat_viewmats, flat_intrinsics = _flatten_camera_matrices(viewmats, intrinsics, num_images)
    camera_to_world = torch.linalg.inv(flat_viewmats)
    depth_normal = depth_to_normal(
        depth,
        valid,
        camera_to_world,
        flat_intrinsics,
        normal_epsilon=config.normal_epsilon,
        orient_toward_camera=True,
    )
    if config.normal_reduction == "mean_hits":
        raise ValueError("mean_hits is unavailable for fused raster outputs")
    normal_consistency = normal_consistency_from_rendered(
        rendered_normals.reshape(num_images, height, width, 3),
        rendered_alphas.reshape(num_images, height, width, 1),
        depth_normal.normals,
        depth_normal.valid_mask,
        reduction=config.normal_reduction,
        normalization_epsilon=config.normal_epsilon,
    )
    return UnbiasedAuxiliaries(
        sequence=sequence,
        samples=samples,
        convergence=convergence,
        corrected_depth=corrected,
        depth_normal=depth_normal,
        normal_consistency=normal_consistency,
    )


def compute_from_gsplat_outputs(
    rendered_alphas: Tensor,
    rendered_normals: Tensor,
    meta: Mapping[str, Any],
    viewmats: Tensor,
    intrinsics: Tensor,
    *,
    scene_radius: float,
    config: UnbiasedSurfelConfig = UnbiasedSurfelConfig(),
) -> UnbiasedAuxiliaries:
    """Production adapter using gsplat's exact primary alpha/normal accumulations."""

    from ._cuda import cuda_available

    if cuda_available() and meta["means2d"].is_cuda:
        return _compute_from_fused_raster(
            rendered_alphas,
            rendered_normals,
            meta,
            viewmats,
            intrinsics,
            scene_radius=scene_radius,
            config=config,
        )
    sequence = collect_ordered_hits(meta)
    return compute_from_hit_sequence(
        meta,
        sequence,
        viewmats,
        intrinsics,
        scene_radius=scene_radius,
        config=config,
        rendered_alphas=rendered_alphas,
        rendered_normals=rendered_normals,
    )


def weighted_regularization(
    auxiliaries: UnbiasedAuxiliaries,
    config: UnbiasedSurfelConfig = UnbiasedSurfelConfig(),
) -> Tensor:
    """Combine only the two new regularizers with explicit paper weights."""

    validate_config(config)
    return (
        config.convergence_weight * auxiliaries.convergence.loss
        + config.normal_weight * auxiliaries.normal_consistency.loss
    )


def prepare_color_target(
    image: Tensor,
    *,
    dataset_kind: str,
    config: UnbiasedSurfelConfig = UnbiasedSurfelConfig(),
) -> Tensor:
    """Apply gamma only to DTU; all other dataset targets are returned unchanged."""

    validate_config(config)
    if dataset_kind.strip().lower() == "dtu":
        return image.clamp_min(0.0).pow(config.dtu_gamma)
    return image


def densification_enabled(
    iteration: int, config: UnbiasedSurfelConfig = UnbiasedSurfelConfig()
) -> bool:
    """The stop iteration is exclusive: iteration 20,000 is already stopped."""

    validate_config(config)
    if iteration < 0:
        raise ValueError("iteration must be nonnegative")
    return iteration < config.densification_stop_iteration


def clamp_tangent_scales_after_densification_(
    scales: Tensor,
    maximum_scale: Tensor | float,
    *,
    iteration: int,
    config: UnbiasedSurfelConfig = UnbiasedSurfelConfig(),
) -> Tensor:
    """Explicit optimizer-step hook for the two 2D surfel tangent scales.

    The operation is in-place under ``no_grad`` and must be called after the
    optimizer update.  ``maximum_scale`` is supplied by the host trainer's
    densification threshold; this adapter does not guess it.
    """

    if scales.shape[-1] < 2:
        raise ValueError("scales must expose at least two tangent axes")
    if densification_enabled(iteration, config):
        return scales
    with torch.no_grad():
        scales[..., :2].clamp_max_(maximum_scale)
    return scales


def clamp_log_tangent_scales_after_densification_(
    log_scales: Tensor,
    maximum_scale: Tensor | float,
    *,
    iteration: int,
    config: UnbiasedSurfelConfig = UnbiasedSurfelConfig(),
) -> Tensor:
    """Clamp an official-gsplat log-scale parameter after densification.

    Official gsplat trainers optimize ``log(scale)`` rather than the physical
    scale.  This hook is the parameter-space counterpart of
    :func:`clamp_tangent_scales_after_densification_`; ``maximum_scale`` is
    still supplied by the host's real densification strategy.
    """

    if log_scales.shape[-1] < 2:
        raise ValueError("log_scales must expose at least two tangent axes")
    if densification_enabled(iteration, config):
        return log_scales
    limit = torch.as_tensor(
        maximum_scale, dtype=log_scales.dtype, device=log_scales.device
    )
    if bool((limit <= 0).any().item()):
        raise ValueError("maximum_scale must be strictly positive")
    with torch.no_grad():
        log_scales[..., :2].clamp_max_(torch.log(limit))
    return log_scales
