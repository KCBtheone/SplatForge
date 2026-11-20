"""Analytic geometric postconditions and an independent rational sign oracle."""
from fractions import Fraction as Q
import numpy as np
import pytest

from splatforge.mesh.processing import (
    TriangleMesh, GeometryError, audit, boundary_loops, cap_boundaries, clip_plane,
    compact, fill_holes, largest_component, orient, refine_face_count,
    repair_self_intersections, self_intersections, smooth_region,
)
from splatforge.mesh import _geometry_native as native


def mesh(v, f):
    return TriangleMesh(np.ascontiguousarray(v, dtype=np.float64), np.ascontiguousarray(f, dtype=np.int64).reshape(-1, 3))


def cube():
    v = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]], float)
    f = [[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]]
    return mesh(v, f)


def sign(x): return (x > 0) - (x < 0)


def determinant(a, b, c, d):
    x = [[Q(float(p[i])) - Q(float(d[i])) for i in range(3)] for p in (a,b,c)]
    return sum(x[0][k] * (x[1][(k+1)%3]*x[2][(k+2)%3]-x[1][(k+2)%3]*x[2][(k+1)%3]) for k in range(3))


def test_exact_predicates_against_fraction_oracle():
    rng = np.random.default_rng(93841)
    for i in range(500):
        points = rng.normal(size=(4,3))
        if i % 2:
            points[3] = points[:3].mean(axis=0)
            points[3,2] = np.nextafter(points[3,2], np.inf)
        if i % 5 == 0:
            points *= 2. ** int(rng.integers(-500,500))
        assert native.orient3d(*points) == sign(determinant(*points))
        a,b,c = points[:3]
        value = (Q(float(a[0]))-Q(float(c[0])))*(Q(float(b[1]))-Q(float(c[1]))) - (Q(float(a[1]))-Q(float(c[1])))*(Q(float(b[0]))-Q(float(c[0])))
        assert native.orient2d(a,b,c) == sign(value)


def test_closed_cube_and_orientation():
    m = cube(); report = audit(m)
    assert report['passed'] and report['euler'] == 2 and report['component_volumes'] == [1.]
    flipped = mesh(m.vertices, m.faces[:,::-1])
    assert not audit(flipped)['outward']
    assert audit(orient(flipped))['passed']
    scrambled = m.faces.copy(); scrambled[[1,3,5]] = scrambled[[1,3,5],::-1]
    assert audit(orient(mesh(m.vertices,scrambled)))['passed']


@pytest.mark.parametrize('normal,offset,volume', [((0,0,1),.25,.75), ((0,0,1),0.,1.), ((1,0,0),.5,.5), ((1,1,1),0.,1.)])
def test_shared_edge_clipping_and_cap(normal, offset, volume):
    m=cube(); result=clip_plane(m, normal, offset)
    assert audit(result)['passed']
    np.testing.assert_array_equal(result.vertices[:len(m.vertices)],m.vertices)
    assert abs(sum(audit(result)['component_volumes'])-volume)<1e-12


def test_empty_clip_and_open_clip():
    assert not len(clip_plane(cube(),offset=2).faces)
    m=clip_plane(cube(),offset=.3,cap=False)
    assert len(boundary_loops(m))==1
    assert audit(cap_boundaries(m,(0,0,1),.3))['passed']


def test_concave_prism_cap():
    xy=np.array([[0,0],[3,0],[3,1],[1,1],[1,3],[0,3]],float)
    n=len(xy);v=np.c_[np.tile(xy,(2,1)),np.repeat([0.,1.],n)]
    faces=[]
    for i in range(n):
        j=(i+1)%n;faces.extend([[i,j,j+n],[i,j+n,i+n]])
    open_mesh=mesh(v,faces)
    top=cap_boundaries(open_mesh,(0,0,-1),-1.)
    closed=cap_boundaries(top,(0,0,1),0.)
    assert audit(closed)['passed']
    assert abs(sum(audit(closed)['component_volumes'])-5)<1e-12


def test_nested_contour_cap_preserves_tube_hole():
    outer=np.array([[0,0],[3,0],[3,3],[0,3]],float)
    inner=np.array([[1,1],[1,2],[2,2],[2,1]],float)
    xy=np.concatenate((outer,inner));v=np.c_[np.tile(xy,(2,1)),np.repeat([0.,1.],8)]
    f=[]
    for loop in (range(4),range(4,8)):
        ids=list(loop)
        for i,a in enumerate(ids):
            b=ids[(i+1)%4];f.extend([[a,b,b+8],[a,b+8,a+8]])
    m=mesh(v,f);m=cap_boundaries(m,(0,0,-1),-1);m=cap_boundaries(m,(0,0,1),0)
    report=audit(m)
    assert report['passed'] and report['euler']==0
    assert abs(sum(report['component_volumes'])-8)<1e-12


def test_fill_hole_preserves_original_coordinates():
    m=cube();opened=mesh(m.vertices,m.faces[2:])
    filled=fill_holes(opened)
    np.testing.assert_array_equal(filled.vertices,m.vertices)
    assert audit(filled)['passed']


@pytest.mark.parametrize('vertices,faces,expected',[
    ([[0,0,0],[2,0,0],[0,2,0],[.2,.2,-1],[.2,.2,1],[1,1,0]],[[0,1,2],[3,4,5]],1),
    ([[0,0,0],[2,0,0],[0,2,0],[.1,.1,0],[1,.1,0],[.1,1,0]],[[0,1,2],[3,4,5]],1),
    ([[0,0,0],[1,0,0],[0,1,0],[0,-1,0]],[[0,1,2],[1,0,3]],0),
    ([[0,0,0],[1,0,0],[0,1,0],[.2,.2,0]],[[0,1,2],[1,0,3]],1),
    ([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],[[0,1,2],[1,0,3]],0),
    ([[0,0,0],[1,0,0],[0,1,0],[1,0,1],[0,1,1]],[[0,1,2],[0,3,4]],0),
    ([[0,0,0],[2,0,0],[0,2,0],[1,1,-1],[1,1,1]],[[0,1,2],[0,3,4]],1),
    ([[0,0,0],[1,0,0],[0,1,0]],[[0,1,2],[0,2,1]],1),
])
def test_triangle_contact_semantics(vertices,faces,expected):
    assert len(self_intersections(mesh(vertices,faces))['pairs'])==expected


def test_nonmanifold_vertex_umbrellas():
    a=cube();b=cube();v=np.r_[a.vertices,b.vertices+np.array([1,1,1])]
    f=b.faces+8;f[f==8]=6 # second cube shares only cube1's upper corner
    result=mesh(v,np.r_[a.faces,f]);r=audit(result,intersections=False)
    assert 6 in r['nonmanifold_vertices'] and not r['passed']


def test_component_selection_and_no_audit_without_intersections():
    a=cube();b=cube();m=mesh(np.r_[a.vertices,b.vertices*.1+3],np.r_[a.faces,b.faces+8])
    assert audit(m)['components']==2
    assert audit(largest_component(m))['component_volumes']==[1.]
    assert not audit(a,intersections=False)['passed']


def test_truncated_intersection_search_cannot_claim_complete():
    m=mesh([[0,0,0],[1,0,0],[0,1,0]],[[0,1,2],[0,2,1],[0,1,2]])
    assert not self_intersections(m,max_pairs=1)['complete']


def test_harmonic_smoothing_locks_exterior():
    m=refine_face_count(cube(),24);free=np.zeros(len(m.vertices),bool);free[8:]=True
    v=m.vertices.copy();v[8:,2]+=.01;m=mesh(v,m.faces)
    assert audit(m)['passed']
    result=smooth_region(m,free,max_displacement=.02)
    np.testing.assert_array_equal(result.vertices[~free],m.vertices[~free])
    assert np.linalg.norm(result.vertices-m.vertices,axis=1).max()<=.0200000000001
    assert audit(result)['passed']


def test_refinement_exact_budget_and_volume():
    result=refine_face_count(cube(),300)
    assert len(result.faces)==300 and audit(result)['passed']
    assert abs(sum(audit(result)['component_volumes'])-1)<1e-12
    with pytest.raises(ValueError):refine_face_count(cube(),13)


def test_repair_fills_existing_hole_and_is_bounded():
    m=cube();result=repair_self_intersections(mesh(m.vertices,m.faces[2:]))
    assert audit(result)['passed']
    bad=mesh([[0,0,0],[1,0,0],[0,1,0]],[[0,1,2],[0,2,1]])
    with pytest.raises(GeometryError):repair_self_intersections(bad,max_removed_faces=1)


def test_input_contract_rejects_implicit_float_cast():
    m=cube()
    with pytest.raises(TypeError):TriangleMesh(m.vertices.astype('f4'),m.faces)
    with pytest.raises(TypeError):TriangleMesh(m.vertices,m.faces.astype('i4'))
    with pytest.raises(ValueError):mesh([[np.nan,0,0]],[])
    with pytest.raises(ValueError):clip_plane(m,(0,0,0))


def test_bvh_matches_exhaustive_pair_enumeration():
    rng=np.random.default_rng(1362);v=rng.normal(size=(45,3));f=np.arange(45).reshape(-1,3)
    m=mesh(v,f);observed=set(map(tuple,self_intersections(m)['pairs']))
    expected=set()
    for a in range(len(f)):
        for b in range(a+1,len(f)):
            if len(self_intersections(mesh(v,f[[a,b]]))['pairs']):expected.add((a,b))
    assert observed==expected
