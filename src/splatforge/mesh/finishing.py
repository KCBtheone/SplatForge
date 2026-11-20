"""Reusable finishing recipe above the mesh operations; no client paths."""
from dataclasses import dataclass
import time
from .processing import (
    GeometryError, audit, clip_plane, fill_holes, largest_component, orient,
    refine_face_count, repair_self_intersections, self_intersections,
    clean_faces, split_vertex_fans,
    repair_contact_stars, TriangleMesh,
    _count_operation,
)


@dataclass(frozen=True)
class FinishConfig:
    cut_normal: tuple[float, float, float] = (0.,0.,1.)
    cut_offset: float | None = None
    keep_largest: bool = True
    repair: bool = True
    repair_iterations: int = 3
    repair_expand_rings: int = 0
    repair_face_budget: int = 10000
    target_faces: int | None = None
    clean_invalid_faces: bool = False
    split_disconnected_fans: bool = False
    hole_method: str = 'projected'
    max_hole_edges: int | None = None
    clip_backend: str = 'cpu'
    repair_strategy: str = 'local_patch'
    pre_cut_offset: float | None = None
    shift_base_to_origin: bool = False
    intersection_backend: str = 'cpu'

    @classmethod
    def native_local(cls, *, initial_cut=0., final_cut=.02, target_faces=300000, clip_backend='cuda', intersection_backend='cpu'):
        """Explicit two-plane local repair recipe; coordinates must be upright."""
        return cls(pre_cut_offset=initial_cut, cut_offset=final_cut, target_faces=target_faces,
            clip_backend=clip_backend, repair_strategy='cut_patch_then_contact_stars',
            repair_expand_rings=1, hole_method='minimal_area', max_hole_edges=2048,
            shift_base_to_origin=True, intersection_backend=intersection_backend)


def finish_mesh(mesh, config=FinishConfig()):
    """Return (accepted mesh, stage report), or raise; never overwrites inputs."""
    _count_operation('public_finish')
    if config.clip_backend not in ('cpu', 'cuda'):
        raise ValueError('clip_backend must be cpu or cuda')
    if config.intersection_backend not in ('cpu', 'cuda_filtered_exact'):
        raise ValueError('unknown intersection backend')
    if config.repair_strategy == 'cut_patch_then_contact_stars':
        return _finish_native_local(mesh, config)
    if config.repair_strategy != 'local_patch':
        raise ValueError('unknown repair strategy')
    if config.pre_cut_offset is not None or config.shift_base_to_origin:
        raise ValueError('two-plane/base-origin options require the native_local strategy')
    stages=[];start=time.perf_counter()

    def stage(name, operation):
        nonlocal mesh
        tick=time.perf_counter()
        try: mesh=operation(mesh)
        except GeometryError as exc: raise GeometryError(f'{name}: {exc}',candidate=exc.candidate,report=exc.report) from exc
        stages.append({'stage':name,'seconds':time.perf_counter()-tick,'vertices':len(mesh.vertices),'faces':len(mesh.faces)})

    if config.clean_invalid_faces: stage('clean_invalid_faces',lambda m:clean_faces(m)[0])
    if config.split_disconnected_fans: stage('split_disconnected_fans',split_vertex_fans)
    if config.keep_largest: stage('largest_component',largest_component)
    stage('consistent_orientation',lambda m:orient(m,outward=False))
    if config.cut_offset is not None:
        if config.clip_backend == 'cuda':
            from .cuda_ops import clip_plane_cuda
            cuda_timing = {}
            def cuda_clip(m):
                result, timing = clip_plane_cuda(m, config.cut_normal, config.cut_offset)
                cuda_timing.update(timing)
                return result
            stage('clip_cap_cuda', cuda_clip)
            stages[-1].update(cuda_timing)
        else:
            stage('clip_cap',lambda m:clip_plane(m,config.cut_normal,config.cut_offset))
        if config.keep_largest: stage('largest_after_clip',largest_component)
    stage('boundary_patches',lambda m:fill_holes(m,check=False,method=config.hole_method,max_edges=config.max_hole_edges))
    # One full check also decides whether repair is needed. Reuse it when no
    # geometry changes afterward instead of searching every triangle pair twice.
    report=audit(mesh, intersection_backend=config.intersection_backend)
    if config.repair and report['self_intersection_pairs']:
        stage('local_repair',lambda m:repair_self_intersections(m,max_iterations=config.repair_iterations,expand_rings=config.repair_expand_rings,max_removed_faces=config.repair_face_budget,hole_method=config.hole_method,max_hole_edges=config.max_hole_edges,keep_largest=config.keep_largest,intersection_backend=config.intersection_backend))
        report=None
    if config.target_faces is not None:
        if len(mesh.faces)>config.target_faces:
            from .simplification import simplify
            stage('qem',lambda m:simplify(m,config.target_faces,intersection_backend=config.intersection_backend))
            report=None
        if len(mesh.faces)<config.target_faces:
            stage('surface_preserving_refinement',lambda m:refine_face_count(m,config.target_faces))
            report=None
    if report is None: report=audit(mesh,intersection_backend=config.intersection_backend)
    if not report['passed'] or (config.keep_largest and report['components']!=1):
        raise GeometryError(f'finishing failed: {report}')
    return mesh,{'stages':stages,'geometry':report,'total_seconds':time.perf_counter()-start}


def _finish_native_local(mesh, config):
    """Fixed two-stage solver, followed by crop/component/budget acceptance.

    The cut/patch prefix may return its bounded intermediate with collisions.
    Topological contact-star repair is always executed. A closed intermediate
    with remaining collisions receives one geometric shared-star stage. Each
    stage retains its own original iteration/removal bound. No intermediate is
    accepted without the final full audit.
    """
    import numpy as np
    if config.pre_cut_offset is None or config.cut_offset is None or not config.repair:
        raise ValueError('native_local requires two cut planes and repair enabled')
    stages = []; started = time.perf_counter(); prefix_status = {}; contact_status = {}; geometric_status = {}
    def stage(name, operation):
        nonlocal mesh
        tick = time.perf_counter()
        try:
            mesh = operation(mesh)
        except GeometryError as exc:
            # Retain the last completed stage for diagnosis/recovery. It is an
            # unaccepted candidate; this exception still aborts finishing.
            candidate = exc.candidate if exc.candidate is not None else mesh
            raise GeometryError(f'{name}: {exc}', candidate=candidate, report=exc.report) from exc
        stages.append({'stage': name, 'seconds': time.perf_counter()-tick,
                       'vertices': len(mesh.vertices), 'faces': len(mesh.faces)})
    def clip(m, offset, cap):
        if config.clip_backend == 'cuda':
            from .cuda_ops import clip_plane_cuda
            return clip_plane_cuda(m, config.cut_normal, offset, cap=cap)[0]
        return clip_plane(m, config.cut_normal, offset, cap=cap)
    def prefix(m):
        try:
            result = repair_self_intersections(m, max_iterations=config.repair_iterations,
                expand_rings=config.repair_expand_rings, max_removed_faces=config.repair_face_budget,
                hole_method=config.hole_method, max_hole_edges=config.max_hole_edges,
                keep_largest=config.keep_largest, intersection_backend=config.intersection_backend)
            prefix_status['partial_intermediate'] = False
            return result
        except GeometryError as exc:
            if exc.candidate is None:
                raise
            prefix_status.update(partial_intermediate=True, remaining_geometry=exc.report)
            return exc.candidate
    def topological_contacts(m):
        try:
            return repair_contact_stars(m, max_iterations=config.repair_iterations,
                expand_rings=config.repair_expand_rings, max_removed_faces=config.repair_face_budget,
                max_hole_edges=config.max_hole_edges, intersection_backend=config.intersection_backend)
        except GeometryError as exc:
            if (exc.candidate is None or exc.report is None
                    or not exc.report.get('watertight_manifold') or not exc.report.get('outward')):
                raise
            contact_status.update(partial_intermediate=True, remaining_geometry=exc.report)
            return exc.candidate
    def geometric_contacts(m):
        try:
            return repair_contact_stars(m, max_iterations=config.repair_iterations,
                expand_rings=config.repair_expand_rings, max_removed_faces=config.repair_face_budget,
                max_hole_edges=config.max_hole_edges, intersection_backend=config.intersection_backend,
                geometric_contacts=True)
        except GeometryError as exc:
            if (exc.candidate is None or exc.report is None
                    or not exc.report.get('watertight_manifold') or not exc.report.get('outward')):
                raise
            geometric_status.update(partial_intermediate=True, remaining_geometry=exc.report)
            return exc.candidate
    if config.clean_invalid_faces:
        stage('clean_invalid_faces', lambda m: clean_faces(m)[0])
    stage('open_trim', lambda m: clip(m, config.pre_cut_offset, False))
    stage('split_vertex_fans', split_vertex_fans)
    if config.keep_largest:
        stage('largest_component', largest_component)
    stage('boundary_patches', lambda m: fill_holes(m, check=False, method=config.hole_method, max_edges=config.max_hole_edges))
    stage('local_patch_prefix', prefix)
    stages[-1].update(prefix_status)
    stage('contact_star_repair', topological_contacts)
    stages[-1].update(contact_status)
    if contact_status:
        stage('geometric_contact_stars', geometric_contacts)
        stages[-1].update(geometric_status)
    stage('final_clip_cap', lambda m: clip(m, config.cut_offset, True))
    if config.keep_largest:
        stage('largest_after_final_clip', largest_component)
    if geometric_status:
        # Repair the retained domain after the fixed crop discards the base band.
        # This second domain must pass; its failure is never deferred.
        stage('retained_geometric_contact_stars', lambda m: repair_contact_stars(m,
            max_iterations=config.repair_iterations, expand_rings=config.repair_expand_rings,
            max_removed_faces=config.repair_face_budget, max_hole_edges=config.max_hole_edges,
            intersection_backend=config.intersection_backend, geometric_contacts=True))
    if config.shift_base_to_origin:
        normal = np.asarray(config.cut_normal, float); normal /= np.linalg.norm(normal)
        stage('base_to_origin', lambda m: TriangleMesh(np.ascontiguousarray(m.vertices-normal*config.cut_offset), m.faces))
    # The final clip and coordinate translation occur after the earlier contact
    # pass. Check that retained domain before changing its triangle count.
    retained = audit(mesh, intersection_backend=config.intersection_backend)
    if retained['outward'] and retained['self_intersection_pairs']:
        stage('post_transform_geometric_contacts', lambda m: repair_contact_stars(m,
            max_iterations=config.repair_iterations, expand_rings=config.repair_expand_rings,
            max_removed_faces=config.repair_face_budget, max_hole_edges=config.max_hole_edges,
            intersection_backend=config.intersection_backend, geometric_contacts=True))
    if config.target_faces is not None:
        if len(mesh.faces) > config.target_faces:
            from .simplification import simplify
            stage('qem', lambda m: simplify(m, config.target_faces, intersection_backend=config.intersection_backend))
        elif len(mesh.faces) < config.target_faces:
            stage('surface_preserving_refinement', lambda m: refine_face_count(m, config.target_faces))
    report = audit(mesh, intersection_backend=config.intersection_backend)
    if not report['passed'] or (config.keep_largest and report['components'] != 1):
        raise GeometryError(f'native_local finishing failed: {report}', candidate=mesh, report=report)
    return mesh, {'stages': stages, 'geometry': report,
                  'repair_strategy': config.repair_strategy, 'total_seconds': time.perf_counter()-started}
