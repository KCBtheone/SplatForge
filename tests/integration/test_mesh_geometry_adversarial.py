"""Adversarial coverage independent of client data or restricted SDKs."""
from fractions import Fraction
import numpy as np
import pytest
from splatforge.mesh import clean_faces, split_vertex_fans, repair_self_intersections, audit, boundary_loops
from splatforge.mesh.processing import _mesh, _triangulate_rings
from splatforge.mesh import _geometry_native as native


def test_dyadic_and_expansion_signs_on_exact_cancellation():
    rng=np.random.default_rng(482109)
    for exponent in [-1000,-500,-200,-100,0,100,200,500,900]:
        for _ in range(30):
            a,b=rng.uniform(.5,1.,size=(2,3))*2.**exponent
            c=a*.25+b*.75
            # All coordinates are the same dyadic affine combination except
            # rounding; evaluate exactly represented inputs independently.
            x=[[Fraction(float(p[i]))-Fraction(float(c[i])) for i in range(2)] for p in (a,b)]
            det=x[0][0]*x[1][1]-x[0][1]*x[1][0]
            assert native.orient2d(a,b,c)==((det>0)-(det<0))


def test_exact_degeneracy_does_not_underflow():
    v=np.array([[0.,0.,0.],[1e-250,0.,0.],[0.,1e-250,0.]])
    f=np.array([[0,1,2]],np.int64)
    assert len(native.degenerate_faces(v,f))==0
    v[2]=[2e-250,0.,0.]
    assert len(native.degenerate_faces(v,f))==1


def test_split_fans_preserves_coordinates_and_reports_contact():
    m=_mesh([[0,0,0],[1,0,0],[0,1,0],[-1,0,0],[0,-1,0]],[[0,1,2],[0,3,4]])
    result=split_vertex_fans(m)
    assert len(result.vertices)==6
    assert len(boundary_loops(result))==2
    np.testing.assert_array_equal(result.vertices[:5],m.vertices)
    assert audit(result)['self_intersection_pairs']==1


def test_clean_faces_is_explicit_and_returns_original_ids():
    m=_mesh([[0,0,0],[1,0,0],[0,1,0]],[[0,1,2],[2,1,0],[0,0,1]])
    result,removed=clean_faces(m)
    assert removed.tolist()==[1,2]
    np.testing.assert_array_equal(result.vertices,m.vertices)
    assert result.faces.tolist()==[[0,1,2]]


def test_actual_self_intersection_repair_not_only_open_hole():
    v=np.array([[1.,1.,1.],[-1.,-1.,1.],[-1.,1.,-1.],[1.,-1.,-1.]])
    f=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],np.int64)
    from splatforge.mesh import orient
    m=orient(_mesh(v,f));bad=_mesh(v,np.r_[m.faces,m.faces[:1]])
    assert audit(bad)['self_intersection_pairs']>0
    repaired=repair_self_intersections(bad)
    assert audit(repaired)['passed']
    np.testing.assert_array_equal(repaired.vertices,bad.vertices)


def test_exact_sign_triangulation_retains_nearly_collinear_vertex():
    # Dyadic samples where a float cross product rounds to zero while its
    # mathematically exact determinant is nonzero. No client geometry is used.
    rng=np.random.default_rng(4013)
    for _ in range(2000):
        a,b=rng.uniform(.1,1.,size=(2,3));a[2]=b[2]=0
        c=(a+b)/2
        cross=np.cross(b-a,c-a)[2]
        if cross==0 and native.orient2d(a,b,c)!=0:break
    else:raise AssertionError('failed to construct independent cancellation fixture')
    sign=native.orient2d(a,b,c)
    v=np.array([a,b,c]);ring=np.array([0,1,2] if sign>0 else [0,2,1],np.int64)
    f=_triangulate_rings(v,[ring],np.array([0.,0.,1.]))
    assert len(f)==1
    assert len(native.degenerate_faces(v,f))==0


def test_minimal_area_nonplanar_patch_and_explicit_bound():
    from splatforge.mesh import fill_holes, orient
    v=np.array([[1.,1.,1.],[-1.,-1.,1.],[-1.,1.,-1.],[1.,-1.,-1.]])
    f=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],np.int64)
    closed=orient(_mesh(v,f));opened=_mesh(v,closed.faces[:-1])
    patched=fill_holes(opened,method='minimal_area',max_edges=16)
    assert audit(patched)['passed']
    with pytest.raises(ValueError):fill_holes(opened,method='minimal_area')


def test_geometric_contact_stars_repair_a_manifold_folded_fan():
    from splatforge.mesh.processing import TriangleMesh, orient, repair_contact_stars
    n = 16
    angle = np.arange(n) * 2 * np.pi / n
    ring = np.c_[np.cos(angle), np.sin(angle)]
    vertices = np.r_[np.c_[ring * 2, np.zeros(n)], np.c_[ring * 2, np.ones(n)],
                     np.c_[ring, np.ones(n)], [[1.5, 0, 1], [0, 0, 0]]]
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.extend([[i, j, n+j], [i, n+j, n+i], [n+i, n+j, 2*n+j],
                      [n+i, 2*n+j, 2*n+i], [2*n+i, 2*n+j, 3*n], [j, i, 3*n+1]])
    folded = orient(TriangleMesh(vertices, np.asarray(faces, dtype=np.int64)))
    before = audit(folded)
    assert before['watertight_manifold'] and not before['nonmanifold_vertices']
    assert before['self_intersection_pairs'] > 0
    repaired = repair_contact_stars(folded, max_iterations=1, max_hole_edges=128,
                                    geometric_contacts=True)
    after = audit(repaired)
    assert after['passed'] and after['components'] == 1 and after['euler'] == 2
    assert np.isclose(sum(after['component_volumes']), 8 * np.sin(np.pi / 8) * 4)


def test_native_finish_repairs_a_contact_introduced_at_final_clip(monkeypatch):
    from splatforge.mesh import FinishConfig, finish_mesh
    from splatforge.mesh import finishing
    from splatforge.mesh.processing import TriangleMesh, orient

    n = 16
    angle = np.arange(n) * 2 * np.pi / n
    ring = np.c_[np.cos(angle), np.sin(angle)]
    vertices = np.r_[np.c_[ring * 2, np.zeros(n)], np.c_[ring * 2, np.ones(n)],
                     np.c_[ring, np.ones(n)], [[0., 0., 1.], [0., 0., 0.]]]
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.extend([[i, j, n+j], [i, n+j, n+i], [n+i, n+j, 2*n+j],
                      [n+i, 2*n+j, 2*n+i], [2*n+i, 2*n+j, 3*n], [j, i, 3*n+1]])
    valid = orient(TriangleMesh(vertices, np.asarray(faces, dtype=np.int64)))
    folded_vertices = vertices.copy()
    folded_vertices[3*n, 0] = 1.5
    folded = orient(TriangleMesh(folded_vertices, np.asarray(faces, dtype=np.int64)))
    assert audit(valid)['passed']
    assert audit(folded)['watertight_manifold'] and audit(folded)['self_intersection_pairs'] > 0

    original_clip = finishing.clip_plane

    def late_contact(mesh, normal, offset, *, cap=True):
        # Exercise a real geometric fold appearing after the early repair pass.
        return folded if cap else original_clip(mesh, normal, offset, cap=False)

    monkeypatch.setattr(finishing, 'clip_plane', late_contact)
    config = FinishConfig.native_local(initial_cut=-.1, final_cut=0., target_faces=120,
                                       clip_backend='cpu', intersection_backend='cpu')
    result, report = finish_mesh(valid, config)
    assert len(result.faces) == 120
    assert report['geometry']['passed'] and report['geometry']['components'] == 1
    assert report['geometry']['intersection_search_complete']
    assert any(stage['stage'] == 'post_transform_geometric_contacts' for stage in report['stages'])
