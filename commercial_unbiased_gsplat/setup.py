#!/usr/bin/env python3
"""Build the optional fused CUDA extension. Apache-2.0, independently written."""

from __future__ import annotations

from setuptools import setup

ext_modules = []
cmdclass = {}
try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    if torch.cuda.is_available():
        ext_modules = [
            CUDAExtension(
                name="unbiased_gsplat_cuda",
                sources=[
                    "csrc/unbiased_fused_bind.cpp",
                    "csrc/unbiased_fused.cu",
                    "csrc/unbiased_aux.cu",
                    "csrc/rasterize_unbiased_aux.cu",
                ],
                extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "-lineinfo"]},
            )
        ]
        cmdclass = {"build_ext": BuildExtension}
except Exception:
    ext_modules = []
    cmdclass = {}

setup(ext_modules=ext_modules, cmdclass=cmdclass)
