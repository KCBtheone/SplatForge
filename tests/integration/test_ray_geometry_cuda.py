from unittest.mock import patch

import pytest
import torch

from splatforge.providers.torch.gaussian.ray_geometry_cuda import (
    regularized_ray_geometry_cuda,
)
from splatforge.studio.renderers import renderers


def test_cpu_rejected_before_loading_extension():
    with patch(
        "splatforge.providers.torch.gaussian.ray_geometry_cuda.extension"
    ) as loader:
        with pytest.raises(ValueError, match="CUDA float32"):
            regularized_ray_geometry_cuda(
                torch.zeros(1, 3),
                torch.zeros(1, 4),
                torch.ones(1, 3),
                torch.eye(4),
                torch.eye(3),
                width=1,
                height=1,
            )
        loader.assert_not_called()


def test_explicit_backend_registration():
    assert callable(renderers.get("gsplat-radegs-cuda-stable-moments"))
    # The stable reference remains available and is never silently replaced.
    assert renderers.get("gsplat-radegs-cuda-stable-moments") is not renderers.get(
        "gsplat-radegs-stable-moments"
    )
    assert callable(renderers.get("cuda-binned-torch-composite"))
    assert callable(renderers.get("cuda-binned-events-composite"))


def test_cuda_binning_rejects_cpu_without_fallback():
    from splatforge.providers.torch.gaussian.tile_binning_cuda import tile_members_cuda

    with pytest.raises(ValueError, match="CUDA float32"):
        tile_members_cuda(
            torch.zeros(1, 4), width=16, height=16, tile_size=16, near=0.01, far=100
        )


def test_cuda_events_reject_cpu_without_fallback():
    from splatforge.providers.torch.gaussian.events_cuda import surface_events_cuda

    with pytest.raises(ValueError, match="CUDA float32"):
        surface_events_cuda(
            *([torch.zeros(1)] * 9), mode="gof", near=0.01, far=100, extent=3
        )


def test_event_backend_rejects_radegs():
    from splatforge.studio.config import RenderConfig

    with pytest.raises(ValueError, match="GOF and AmbiSuR"):
        renderers.get("cuda-binned-events-composite")(RenderConfig(), mode="radegs")


@pytest.mark.parametrize("all_padded", [False, True])
def test_fixed_size_absolute_gradient_scatter(all_padded):
    from splatforge.studio.batched_renderer import _accumulate_absolute_gradient

    ids = torch.tensor([[2, 0, 2], [1, 0, 0]])
    valid = torch.tensor([[True, False, True], [True, False, False]])
    if all_padded:
        valid.zero_()
    value = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    # Inactive values must not poison the target, unlike multiplication by0.
    value[~valid] = float("nan")
    expected = torch.zeros(3, 2)
    expected.index_add_(0, ids[valid], value[valid])
    actual = torch.zeros_like(expected)
    _accumulate_absolute_gradient(actual, ids, valid, value)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires actual CUDA")
def test_cuda_event_vjp_and_strided_inputs():
    from tools.surface_cuda_events_check import check

    for mode in ["gof", "ambisur"]:
        check(mode, 5, strided=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires actual CUDA")
def test_cuda_tile_edges_and_stable_ties():
    import numpy as np
    from tools.surface_cuda_binning_check import bins_check

    bounds = np.array([[0, 0, 0, 2], [16, 16, 0, 2], [32, 32, 16, 2]], dtype=np.float32)
    bins_check(bounds)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires actual CUDA")
def test_cuda_truncation_support_matches_reference_order():
    from tools.surface_cuda_events_check import check_truncation_boundary

    check_truncation_boundary()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires actual CUDA")
def test_cuda_trained_gof_quadratic_contraction():
    from tools.surface_cuda_events_check import check_trained_gof_quadratic

    check_trained_gof_quadratic()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires actual CUDA")
def test_cuda_trained_gof_background_event():
    from tools.surface_cuda_events_check import check_trained_gof_background_event

    check_trained_gof_background_event()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires actual CUDA")
def test_cuda_gof_indexed_inverse_adjoint_paths():
    from tools.surface_cuda_events_check import check_indexed_gof_q3

    check_indexed_gof_q3()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires actual CUDA")
def test_cuda_value_gradient_and_mask():
    from tools.surface_ray_geometry_cuda_check import check

    check(33)
    check(33, cameras=True)
    check(33, extreme=True)
