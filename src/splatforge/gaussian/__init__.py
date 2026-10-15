"""Experimental Gaussian geometry operators; no training or optional CUDA imports."""

from .ops import SurfaceOperators, create_surface_runtime

__all__ = ["SurfaceOperators", "create_surface_runtime"]
