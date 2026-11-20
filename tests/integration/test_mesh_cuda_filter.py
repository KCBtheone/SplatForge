"""Classification parity, including contacts that must never be pruned."""
import numpy as np
from splatforge.mesh import processing as p


def cases():
    yield p._mesh([[0,0,0],[1,0,0],[0,1,0],[.1,.1,-1],[.1,.1,1],[.8,.1,0]],[[0,1,2],[3,4,5]])
    yield p._mesh([[0,0,0],[1,0,0],[0,1,0],[0,-1,0],[.2,.2,0]],[[0,1,2],[1,0,3],[0,1,4]])
    # Coincident unshared endpoints must fail even when triangles look adjacent.
    yield p._mesh([[0,0,0],[1,0,0],[0,1,0],[0,0,0],[-1,0,0],[0,-1,0]],[[0,1,2],[3,4,5]])
    yield p._mesh([[0,0,0],[1,0,0],[0,1,0]],[[0,1,2],[2,1,0]])
    rng=np.random.default_rng(482109)
    for exponent in [-1000,-500,0,500,900]:
        v=rng.uniform(-1,1,(30,3))*2.**exponent
        yield p._mesh(v,np.arange(30).reshape(-1,3))


def test_broad_phase_plus_unchanged_exact_refinement_matches_cpu():
    import pytest
    native=pytest.importorskip('splatforge.mesh._geometry_native')
    if not hasattr(native,'broad_pairs'):pytest.skip('rebuild owned native extension')
    for mesh in cases():
        pairs=native.broad_pairs(mesh.vertices,mesh.faces,20000)
        result=native.exact_pairs(mesh.vertices,mesh.faces,pairs,0)
        assert result['pairs'].tolist()==p.self_intersections(mesh)['pairs'].tolist()
    with pytest.raises(RuntimeError,match='budget'):
        native.broad_pairs(mesh.vertices,mesh.faces,1)


def test_cuda_filter_preserves_collision_classification():
    import pytest
    cp=pytest.importorskip('cupy')
    try: cp.cuda.runtime.getDeviceCount()
    except Exception:pytest.skip('CUDA device unavailable')
    for mesh in cases():
        expected=p.self_intersections(mesh)['pairs'].tolist()
        result=p.self_intersections(mesh,backend='cuda_filtered_exact')
        assert result['complete'] and result['pairs'].tolist()==expected
    before=p.activation_counts().get('self_intersections_cpu',0)
    p.self_intersections(next(cases()),backend='cuda_filtered_exact')
    assert p.activation_counts().get('self_intersections_cpu',0)==before
