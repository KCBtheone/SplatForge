from unittest.mock import patch

import pytest
import torch

from splatforge.gaussian import SurfaceOperators
from splatforge.core.errors import SchemaValidationError


@pytest.mark.parametrize("requested", ["none", "l1", "l2", "both"])
def test_selected_composite_preserves_requested_values_and_adjoints(requested):
    ops = SurfaceOperators()
    torch.manual_seed(93)
    alpha = (torch.rand(2, 5, 7, dtype=torch.float64) * 0.3).requires_grad_()
    color = torch.rand(2, 5, 3, dtype=torch.float64, requires_grad=True)
    depth = torch.rand(2, 5, 7, dtype=torch.float64, requires_grad=True)
    normal = torch.rand(2, 5, 7, 3, dtype=torch.float64, requires_grad=True)
    inputs = (alpha, color, depth, normal)
    full = ops("splat.alpha_composite@2", *inputs)
    selected = ops("splat.alpha_composite@3", *inputs, distortions=requested)
    keep = [0, 1, 2, 3, 6, 7]
    if requested in ("l1", "both"):
        keep.append(4)
    if requested in ("l2", "both"):
        keep.append(5)
    for index in keep:
        torch.testing.assert_close(full[index], selected[index], rtol=0, atol=0)
    a = torch.autograd.grad(sum(full[i].sum() for i in keep), inputs, retain_graph=True)
    b = torch.autograd.grad(sum(selected[i].sum() for i in keep), inputs)
    for first, second in zip(a, b):
        torch.testing.assert_close(first, second, rtol=1e-12, atol=1e-12)


def test_no_distortions_never_sorts_depths_and_requires_explicit_selection():
    ops = SurfaceOperators()
    values = (
        torch.ones(1, 2, 3) * 0.2,
        torch.ones(1, 2, 3),
        torch.ones(1, 2, 3),
        torch.ones(1, 2, 3, 3),
    )
    with patch.object(
        torch.Tensor, "sort", side_effect=AssertionError("unused depth sorting")
    ):
        result = ops("splat.alpha_composite@3", *values, distortions="none")
    assert result[4].count_nonzero() == 0
    assert result[5].count_nonzero() == 0
    with pytest.raises(SchemaValidationError):
        ops("splat.alpha_composite@3", *values)
