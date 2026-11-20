"""Owned CPU quadric simplification with explicit geometric acceptance."""
import numpy as np
from .processing import GeometryError, _mesh, _native, audit


def simplify(mesh, target_faces, *, locked=None, max_quadric_error=float('inf'), intersection_backend='cpu'):
    """Collapse manifold edges while protecting locked and boundary vertices.

    A target that cannot be attained under the link/normal/error constraints
    raises. Output is accepted only after exhaustive exact intersection checks.
    max_quadric_error bounds the accumulated unweighted squared plane-distance
    quadric, not a Hausdorff distance. Original vertex indexing is retained.
    """
    if not isinstance(target_faces, (int, np.integer)) or not 0 < target_faces <= len(mesh.faces):
        raise ValueError('target_faces must be a positive integer <= current faces')
    if not max_quadric_error >= 0:
        raise ValueError('max_quadric_error must be nonnegative')
    if locked is None: locked=np.zeros(len(mesh.vertices),bool)
    if not isinstance(locked,np.ndarray) or locked.dtype!=bool or locked.shape!=(len(mesh.vertices),) or not locked.flags.c_contiguous:
        raise TypeError('locked must be a contiguous NumPy bool vertex mask')
    if not audit(mesh,intersection_backend=intersection_backend)['passed']:
        raise GeometryError('simplification requires accepted closed input geometry')
    result=_native().simplify(mesh.vertices,mesh.faces,int(target_faces),locked,float(max_quadric_error))
    if not result['target_reached']:
        raise GeometryError(f"QEM stopped at {len(result['faces'])} faces under the requested constraints (target {target_faces})")
    candidate=_mesh(result['vertices'],result['faces'])
    if not np.array_equal(candidate.vertices[locked],mesh.vertices[locked]):
        raise GeometryError('locked vertex changed')
    report=audit(candidate,intersection_backend=intersection_backend)
    if not report['passed']:
        raise GeometryError(f'QEM candidate rejected by full geometry audit: {report}')
    return candidate
