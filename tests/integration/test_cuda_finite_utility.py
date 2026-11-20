from types import SimpleNamespace

import pytest
import torch

from splatforge.providers.torch.gaussian import finite_cuda


def test_explicit_finite_wrapper_preserves_views_and_flag(monkeypatch):
    # Dispatch-only CPU fixture, never a CPU substitute for the CUDA predicate.
    first = torch.ones(2, 7).T
    second = torch.ones(3).expand(5, 3)
    flag = object()

    def check(operands):
        assert len(operands) == 2
        assert operands[0] is first and operands[1] is second
        return flag

    monkeypatch.setattr(finite_cuda, "_extension", SimpleNamespace(all_finite=check))
    assert finite_cuda.cuda_all_finite(iter((first, second))) is flag


def test_finite_extension_requires_owned_cache(monkeypatch):
    monkeypatch.setattr(finite_cuda, "_extension", None)
    monkeypatch.delenv("TORCH_EXTENSIONS_DIR", raising=False)
    with pytest.raises(RuntimeError, match="task-owned"):
        finite_cuda.cuda_all_finite([])


def test_finite_backend_errors_are_not_fallback(monkeypatch):
    def rejected(_):
        raise RuntimeError("explicit CUDA required")

    monkeypatch.setattr(finite_cuda, "_extension", SimpleNamespace(all_finite=rejected))
    with pytest.raises(RuntimeError, match="explicit CUDA"):
        finite_cuda.cuda_all_finite([torch.ones(3)])
