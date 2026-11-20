"""Typed results for the commercial Unbiased Surfel adapter."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class HitSequence:
    """Accepted gsplat hits, grouped by dense image/pixel segment."""

    gaussian_ids: Tensor
    pixel_ids: Tensor
    image_ids: Tensor
    segment_ids: Tensor
    num_images: int
    image_height: int
    image_width: int

    @property
    def num_hits(self) -> int:
        return int(self.gaussian_ids.numel())

    @property
    def num_segments(self) -> int:
        return self.num_images * self.image_height * self.image_width


@dataclass(frozen=True)
class GatheredHitAttributes:
    means2d: Tensor
    ray_transforms: Tensor
    opacities: Tensor
    center_depths: Tensor
    normals: Tensor | None


@dataclass(frozen=True)
class RaySplatSamples:
    """Per-hit Gaussian value and z-depth, with an explicit validity mask."""

    gaussian_weight: Tensor
    depth: Tensor
    rho3: Tensor
    rho2: Tensor
    local_uv: Tensor
    ray_plane_mask: Tensor
    valid_mask: Tensor


@dataclass(frozen=True)
class ConvergenceResult:
    loss: Tensor
    numerator: Tensor
    denominator: Tensor
    per_pair: Tensor
    pair_mask: Tensor
    previous_indices: Tensor
    current_indices: Tensor
    per_segment: Tensor


@dataclass(frozen=True)
class CorrectedDepthResult:
    depth: Tensor
    valid_mask: Tensor
    selected_hit_indices: Tensor
    contribution: Tensor
    cumulative_contribution: Tensor


@dataclass(frozen=True)
class DepthNormalResult:
    points: Tensor
    normals: Tensor
    valid_mask: Tensor


@dataclass(frozen=True)
class CorrectedSurfaceResult:
    """Dense corrected-depth surface without retained per-hit work buffers.

    This compact result is used by the one-sided multi-view neighbor path.  It
    deliberately retains only the dense world points and their authoritative
    validity mask so the much larger ordered-hit tensors can be released before
    the pairwise PGSR-style constraint is evaluated.
    """

    points: Tensor
    valid_mask: Tensor
    num_hits: int
    corrected_valid_pixels: int


@dataclass(frozen=True)
class NormalConsistencyResult:
    loss: Tensor
    numerator: Tensor
    denominator: Tensor
    per_hit: Tensor
    blending_weights: Tensor
    hit_mask: Tensor
    per_segment: Tensor


@dataclass(frozen=True)
class DenseNormalConsistencyResult:
    """Normal loss reconstructed exactly from the primary renderer outputs."""

    loss: Tensor
    numerator: Tensor
    denominator: Tensor
    per_segment: Tensor
    valid_mask: Tensor


@dataclass(frozen=True)
class MultiViewConsistencyResult:
    """Paper-derived PGSR-style pairwise consistency diagnostics.

    The result keeps both explicit reductions and the sampled pixel-domain
    evidence used by the two losses.  It is intentionally independent of any
    host trainer so the geometric contract can be tested on CPU fixtures.
    """

    geometry_loss: Tensor
    photometric_loss: Tensor
    geometry_numerator: Tensor
    geometry_denominator: Tensor
    photometric_numerator: Tensor
    photometric_denominator: Tensor
    reprojection_error: Tensor
    occlusion_weight: Tensor
    geometry_valid_mask: Tensor
    photometric_valid_mask: Tensor
    reference_pixels: Tensor
    neighbor_pixels: Tensor
