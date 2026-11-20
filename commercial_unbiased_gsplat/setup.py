#!/usr/bin/env python3
"""Build the optional fused CUDA extension. Apache-2.0, independently written."""

from __future__ import annotations

import os

from setuptools import setup

build_cuda = os.getenv("UNBIASED_GSPLAT_BUILD_CUDA", "auto")
if build_cuda not in {"0", "1", "auto"}:
    raise ValueError("UNBIASED_GSPLAT_BUILD_CUDA must be 0, 1, or auto")

ext_modules = []
cmdclass = {}
if build_cuda != "0":
    import torch

    if build_cuda == "1" or torch.cuda.is_available():
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension

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
setup(ext_modules=ext_modules, cmdclass=cmdclass)
