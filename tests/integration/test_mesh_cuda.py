"""Actual CUDA clipping checks. No CPU substitution for missing hardware."""
import numpy as np
import pytest

from splatforge.mesh.processing import _mesh, audit
from splatforge.mesh.cuda_ops import clip_plane_cuda


@pytest.fixture(scope='module')
def cuda():
    cp = pytest.importorskip('cupy')
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip('CUDA hardware unavailable')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('CUDA runtime unavailable')
    return cp


def cube():
    return _mesh([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],
                 [[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])


@pytest.mark.parametrize('normal,offset,volume', [((0,0,1),.25,.75),((1,0,0),.5,.5),((0,0,1),0.,1.),((1,1,1),0.,1.)])
def test_cuda_clip_closed_volume(cuda,normal,offset,volume):
    result,timing=clip_plane_cuda(cube(),normal,offset)
    report=audit(result)
    assert report['passed']
    assert abs(sum(report['component_volumes'])-volume)<1e-10
    assert timing['cuda_clip_and_transfer_seconds']>=0


def test_cuda_clip_empty(cuda):
    result,_=clip_plane_cuda(cube(),offset=2.)
    assert not len(result.faces)
