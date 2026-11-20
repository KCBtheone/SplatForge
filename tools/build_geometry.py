"""Build the optional owned CPU geometry extension, without rebuilding Torch.

Run with a Python environment containing setuptools and pybind11. On Windows
use a Visual Studio developer shell (or installed Build Tools discovery).
"""
from pathlib import Path
import sys
from setuptools import Distribution
from pybind11.setup_helpers import Pybind11Extension, build_ext

root = Path(__file__).resolve().parents[1]
distribution = Distribution({
    "name": "splatforge-geometry-native",
    "ext_modules": [Pybind11Extension(
        "splatforge.mesh._geometry_native",
        [str(root / "src/splatforge/mesh/native/geometry.cpp")],
        cxx_std=17,
        depends=[str(p) for p in (root / "src/splatforge/mesh/native").glob("*.h")],
        extra_compile_args=["/fp:strict", "/utf-8"] if sys.platform == "win32" else ["-fno-fast-math", "-ffp-contract=off"],
    )],
    "cmdclass": {"build_ext": build_ext},
})
command = distribution.get_command_obj("build_ext")
command.build_lib = str(root / "src")
command.build_temp = str(root / "build/geometry")
distribution.run_command("build_ext")
