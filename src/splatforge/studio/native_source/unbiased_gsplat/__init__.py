"""Commercial-candidate, source-isolated Unbiased Surfel operators."""

from .convergence import (
    asymmetric_squared_difference,
    depth_convergence,
    segmented_adjacent_pairs,
)
from .corrected_depth import corrected_depth, segmented_cumulative_sum
from .depth_normal import depth_to_normal, depth_to_points
from .hit_sequence import (
    GSPLAT_COMMIT,
    UnsupportedGsplatLayoutError,
    center_depth_order_violations,
    collect_ordered_hits,
    gather_hit_attributes,
    make_hit_sequence,
)
from .losses import (
    blending_weights,
    normal_consistency_from_rendered,
    normal_consistency_loss,
)
from .multiview import (
    PGSRMultiViewConfig,
    apply_homography,
    build_camera_neighbor_graph,
    multiview_consistency,
    normalize_rendered_plane_normals,
    normalized_cross_correlation,
    plane_induced_homography,
    validate_multiview_config,
    weighted_multiview_regularization,
)
from ._cuda import cuda_available, get_build_info, ray_splat_depth_and_weight_fused
from .ray_splat_depth import ray_splat_depth_and_weight, ray_splat_depth_and_weight_torch
from .trainer_adapter import (
    UnbiasedAuxiliaries,
    UnbiasedSurfelConfig,
    clamp_tangent_scales_after_densification_,
    clamp_log_tangent_scales_after_densification_,
    compute_corrected_surface_chunked_from_gsplat_meta,
    compute_corrected_surface_chunked_from_hit_sequence,
    compute_corrected_surface_from_gsplat_meta,
    compute_corrected_surface_from_hit_sequence,
    compute_from_gsplat_meta,
    compute_from_gsplat_outputs,
    compute_from_hit_sequence,
    densification_enabled,
    prepare_color_target,
    weighted_regularization,
)

__all__ = [
    "GSPLAT_COMMIT",
    "PGSRMultiViewConfig",
    "UnbiasedAuxiliaries",
    "UnbiasedSurfelConfig",
    "UnsupportedGsplatLayoutError",
    "asymmetric_squared_difference",
    "apply_homography",
    "blending_weights",
    "build_camera_neighbor_graph",
    "center_depth_order_violations",
    "clamp_tangent_scales_after_densification_",
    "clamp_log_tangent_scales_after_densification_",
    "collect_ordered_hits",
    "compute_corrected_surface_chunked_from_gsplat_meta",
    "compute_corrected_surface_chunked_from_hit_sequence",
    "compute_corrected_surface_from_gsplat_meta",
    "compute_corrected_surface_from_hit_sequence",
    "compute_from_gsplat_meta",
    "compute_from_hit_sequence",
    "corrected_depth",
    "compute_from_gsplat_outputs",
    "densification_enabled",
    "depth_convergence",
    "depth_to_normal",
    "depth_to_points",
    "gather_hit_attributes",
    "make_hit_sequence",
    "multiview_consistency",
    "normal_consistency_loss",
    "normalize_rendered_plane_normals",
    "normalized_cross_correlation",
    "plane_induced_homography",
    "prepare_color_target",
    "cuda_available",
    "get_build_info",
    "ray_splat_depth_and_weight",
    "ray_splat_depth_and_weight_fused",
    "ray_splat_depth_and_weight_torch",
    "segmented_adjacent_pairs",
    "segmented_cumulative_sum",
    "validate_multiview_config",
    "normal_consistency_from_rendered",
    "weighted_multiview_regularization",
    "weighted_regularization",
]
