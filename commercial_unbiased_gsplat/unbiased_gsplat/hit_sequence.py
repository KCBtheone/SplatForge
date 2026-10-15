"""Narrow adapter around the Apache-2.0 gsplat 2DGS hit-index API.

Only unpacked gsplat metadata is accepted in the first implementation phase.  The
CUDA routine remains responsible for tile ordering, ray/splat validity, alpha
cutoff, and transmittance termination; this module never reconstructs or sorts
the candidate set itself.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from ._validation import (
    require_nonnegative_integer_ids,
    require_same_device,
    require_same_length,
    require_sorted_segment_ids,
)
from .models import GatheredHitAttributes, HitSequence


GSPLAT_COMMIT = "6f378361f32ba4f1be86f65ae0db4948efb37dd5"


class UnsupportedGsplatLayoutError(RuntimeError):
    """Raised when metadata uses a gsplat layout not frozen by this adapter."""


def _required(meta: Mapping[str, Any], key: str) -> Any:
    if key not in meta:
        raise KeyError(f"gsplat metadata is missing required key {key!r}")
    return meta[key]


def make_hit_sequence(
    gaussian_ids: Tensor,
    pixel_ids: Tensor,
    image_ids: Tensor,
    *,
    num_images: int,
    image_height: int,
    image_width: int,
    verify_order: bool = True,
) -> HitSequence:
    """Build and validate a dense-pixel hit sequence without changing its order."""

    if num_images <= 0 or image_height <= 0 or image_width <= 0:
        raise ValueError("num_images, image_height, and image_width must be positive")
    require_same_length(
        (
            ("gaussian_ids", gaussian_ids),
            ("pixel_ids", pixel_ids),
            ("image_ids", image_ids),
        )
    )
    require_same_device(
        (
            ("gaussian_ids", gaussian_ids),
            ("pixel_ids", pixel_ids),
            ("image_ids", image_ids),
        )
    )
    require_nonnegative_integer_ids("gaussian_ids", gaussian_ids)
    require_nonnegative_integer_ids("pixel_ids", pixel_ids)
    require_nonnegative_integer_ids("image_ids", image_ids)
    if pixel_ids.numel() and int(pixel_ids.max().item()) >= image_height * image_width:
        raise ValueError("pixel_ids contains an id outside the image")
    if image_ids.numel() and int(image_ids.max().item()) >= num_images:
        raise ValueError("image_ids contains an id outside num_images")

    segment_ids = image_ids.to(torch.int64) * (image_height * image_width)
    segment_ids = segment_ids + pixel_ids.to(torch.int64)
    if verify_order:
        require_sorted_segment_ids(
            segment_ids, num_segments=num_images * image_height * image_width
        )
    return HitSequence(
        gaussian_ids=gaussian_ids.to(torch.int64),
        pixel_ids=pixel_ids.to(torch.int64),
        image_ids=image_ids.to(torch.int64),
        segment_ids=segment_ids,
        num_images=num_images,
        image_height=image_height,
        image_width=image_width,
    )


def collect_ordered_hits(
    meta: Mapping[str, Any],
    *,
    range_start: int = 0,
    range_end: int = 2**31 - 1,
    transmittances: Tensor | None = None,
    verify_order: bool = True,
) -> HitSequence:
    """Ask gsplat for exactly the accepted unpacked 2DGS hits.

    ``range_end`` is deliberately finite and inside a signed 32-bit range even
    though gsplat documents using a very large value for one-shot traversal.
    """

    if range_start < 0 or range_end <= range_start:
        raise ValueError("expected 0 <= range_start < range_end")

    means2d = _required(meta, "means2d")
    ray_transforms = _required(meta, "ray_transforms")
    opacities = _required(meta, "opacities")
    isect_offsets = _required(meta, "isect_offsets")
    flatten_ids = _required(meta, "flatten_ids")
    width = int(_required(meta, "width"))
    height = int(_required(meta, "height"))
    tile_size = int(_required(meta, "tile_size"))

    if not isinstance(means2d, Tensor) or not isinstance(ray_transforms, Tensor):
        raise TypeError("gsplat means2d and ray_transforms must be tensors")
    if means2d.ndim < 3 or ray_transforms.ndim < 4:
        raise UnsupportedGsplatLayoutError(
            "packed=True metadata is not supported in phase 1; render with packed=False"
        )
    if meta.get("camera_ids") is not None or meta.get("gaussian_ids") is not None:
        raise UnsupportedGsplatLayoutError(
            "packed=True metadata is not supported in phase 1; render with packed=False"
        )
    if means2d.shape[-1] != 2 or ray_transforms.shape[-2:] != (3, 3):
        raise ValueError("unexpected gsplat means2d/ray_transforms schema")
    if means2d.shape[:-1] != ray_transforms.shape[:-2]:
        raise ValueError("gsplat means2d and ray_transforms shapes disagree")
    if opacities.shape != means2d.shape[:-1]:
        raise ValueError("gsplat opacities shape must equal means2d.shape[:-1]")
    if tile_size <= 0 or width <= 0 or height <= 0:
        raise ValueError("invalid image or tile dimensions in gsplat metadata")

    image_dims = tuple(means2d.shape[:-2])
    num_images = 1
    for dimension in image_dims:
        num_images *= int(dimension)
    expected_transmittance_shape = image_dims + (height, width)
    if transmittances is None:
        transmittances = torch.ones(
            expected_transmittance_shape,
            dtype=opacities.dtype,
            device=opacities.device,
        )
    elif tuple(transmittances.shape) != expected_transmittance_shape:
        raise ValueError(
            "transmittances must have shape "
            f"{expected_transmittance_shape}, got {tuple(transmittances.shape)}"
        )

    try:
        from gsplat.cuda._wrapper import rasterize_to_indices_in_range_2dgs
    except ImportError as exc:
        raise ImportError(
            "collect_ordered_hits requires gsplat built from commit " + GSPLAT_COMMIT
        ) from exc

    gaussian_ids, pixel_ids, image_ids = rasterize_to_indices_in_range_2dgs(
        range_start,
        range_end,
        transmittances,
        means2d,
        ray_transforms,
        opacities,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
    )
    return make_hit_sequence(
        gaussian_ids,
        pixel_ids,
        image_ids,
        num_images=num_images,
        image_height=height,
        image_width=width,
        verify_order=verify_order,
    )


def gather_hit_attributes(
    meta: Mapping[str, Any], sequence: HitSequence
) -> GatheredHitAttributes:
    """Gather differentiable per-hit tensors from unpacked gsplat metadata."""

    means2d = _required(meta, "means2d")
    ray_transforms = _required(meta, "ray_transforms")
    opacities = _required(meta, "opacities")
    center_depths = _required(meta, "depths")
    normals = meta.get("normals")
    if means2d.ndim < 3 or ray_transforms.ndim < 4:
        raise UnsupportedGsplatLayoutError("gather_hit_attributes requires packed=False")
    num_gaussians = int(means2d.shape[-2])
    flat_means2d = means2d.reshape(-1, num_gaussians, 2)
    flat_transforms = ray_transforms.reshape(-1, num_gaussians, 3, 3)
    flat_opacities = opacities.reshape(-1, num_gaussians)
    flat_depths = center_depths.reshape(-1, num_gaussians)
    if int(flat_means2d.shape[0]) != sequence.num_images:
        raise ValueError("metadata image count does not match the hit sequence")
    image_ids = sequence.image_ids
    gaussian_ids = sequence.gaussian_ids
    if gaussian_ids.numel() and int(gaussian_ids.max().item()) >= num_gaussians:
        raise ValueError("hit sequence contains a Gaussian id outside metadata")

    gathered_normals = None
    if normals is not None:
        if normals.shape[-2:] != (num_gaussians, 3):
            raise ValueError("unexpected gsplat normals shape")
        gathered_normals = normals.reshape(-1, num_gaussians, 3)[
            image_ids, gaussian_ids
        ]
    return GatheredHitAttributes(
        means2d=flat_means2d[image_ids, gaussian_ids],
        ray_transforms=flat_transforms[image_ids, gaussian_ids],
        opacities=flat_opacities[image_ids, gaussian_ids],
        center_depths=flat_depths[image_ids, gaussian_ids],
        normals=gathered_normals,
    )


def center_depth_order_violations(
    center_depths: Tensor, sequence: HitSequence, *, tolerance: float = 0.0
) -> Tensor:
    """Return current-hit indices that violate front-to-back center sorting."""

    if center_depths.shape != (sequence.num_hits,):
        raise ValueError("center_depths must have shape [M]")
    if tolerance < 0.0:
        raise ValueError("tolerance must be nonnegative")
    if sequence.num_hits < 2:
        return torch.empty(0, dtype=torch.int64, device=center_depths.device)
    same_segment = sequence.segment_ids[1:] == sequence.segment_ids[:-1]
    violates = same_segment & (center_depths[1:] + tolerance < center_depths[:-1])
    return torch.nonzero(violates, as_tuple=False).flatten() + 1
