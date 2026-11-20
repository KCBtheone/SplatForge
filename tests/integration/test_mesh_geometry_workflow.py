"""Geometry recipe, optional native QEM and precision file round trips."""
import numpy as np
import pytest
from splatforge.mesh import audit, FinishConfig, finish_mesh, refine_face_count
from splatforge.mesh.formats import read_mesh, write_mesh
from splatforge.mesh.simplification import simplify
from splatforge.mesh.processing import GeometryError, _mesh


def sphere(subdivisions=2):
    v=np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]],float)
    f=np.array([[0,2,4],[2,1,4],[1,3,4],[3,0,4],[2,0,5],[1,2,5],[3,1,5],[0,3,5]],np.int64)
    for _ in range(subdivisions):
        points=v.tolist(); edges={};faces=[]
        def mid(a,b):
            key=tuple(sorted((int(a),int(b))))
            if key not in edges:
                p=v[a]+v[b];p/=np.linalg.norm(p);edges[key]=len(points);points.append(p.tolist())
            return edges[key]
        for a,b,c in f:
            ab,bc,ca=mid(a,b),mid(b,c),mid(c,a)
            faces.extend([[a,ab,ca],[ab,b,bc],[ca,bc,c],[ab,bc,ca]])
        v=np.array(points);f=np.array(faces,np.int64)
    return _mesh(v,f)


def test_owned_qem_preserves_geometry_and_locked_vertices():
    m=sphere();locked=np.zeros(len(m.vertices),bool);locked[:6]=True
    result=simplify(m,64,locked=locked)
    assert len(result.faces)==64 and audit(result)['passed']
    np.testing.assert_array_equal(result.vertices[locked],m.vertices[locked])


def test_qem_rejects_unreachable_target_and_error_bound():
    m=sphere()
    with pytest.raises(GeometryError):simplify(m,64,locked=np.ones(len(m.vertices),bool))
    with pytest.raises(GeometryError):simplify(m,64,max_quadric_error=0.)


@pytest.mark.parametrize('extension',['ply','off'])
def test_precision_roundtrip(tmp_path,extension):
    m=sphere();path=tmp_path/f'mesh.{extension}'
    write_mesh(path,m);out=read_mesh(path)
    np.testing.assert_array_equal(out.vertices,m.vertices)
    np.testing.assert_array_equal(out.faces,m.faces)
    assert audit(out)['passed']


def test_finishing_end_to_end(tmp_path):
    m=sphere();m=_mesh(m.vertices,m.faces[:-1])
    result,report=finish_mesh(m,FinishConfig(cut_offset=.123,target_faces=100))
    assert report['geometry']['passed'] and len(result.faces)==100
    path=tmp_path/'result.ply';write_mesh(path,result)
    assert audit(read_mesh(path))['passed']


def test_malformed_file_rejected(tmp_path):
    path=tmp_path/'bad.ply'
    path.write_bytes(b'ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n0 1 0\n4 0 1 2 3\n')
    with pytest.raises(ValueError):read_mesh(path)
