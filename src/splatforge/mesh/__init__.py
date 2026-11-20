"""Representation-neutral extraction, precision I/O and geometry processing.

Importing the domain does not build or load a native extension. Exact geometry
audits require the optional geometry provider; see docs/mesh/PROCESSING.md.
"""

from .processing import (
    GeometryError, TriangleMesh, audit, boundary_loops, cap_boundaries, clip_plane,
    clean_faces, compact, fill_holes, largest_component, orient, refine_face_count,
    repair_self_intersections, self_intersections, smooth_region,
    split_vertex_fans,
    repair_contact_stars,
)
from .finishing import FinishConfig, finish_mesh
from .simplification import simplify
from .cuda_ops import clip_plane_cuda

__all__ = [
    'GeometryError', 'TriangleMesh', 'FinishConfig', 'finish_mesh', 'audit',
    'boundary_loops', 'cap_boundaries', 'clip_plane', 'compact', 'fill_holes',
    'largest_component', 'orient', 'refine_face_count',
    'repair_self_intersections', 'self_intersections', 'smooth_region',
    'clean_faces', 'split_vertex_fans', 'simplify',
    'clip_plane_cuda',
    'repair_contact_stars',
]
