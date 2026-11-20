"""Optional CPU geometry provider; base installation remains compiler-free."""
import os
import sys
from setuptools import setup

extensions=[]
if os.environ.get('SPLATFORGE_BUILD_GEOMETRY')=='1':
    from pybind11.setup_helpers import Pybind11Extension
    extensions=[Pybind11Extension(
        'splatforge.mesh._geometry_native',
        ['src/splatforge/mesh/native/geometry.cpp'],cxx_std=17,
        depends=['src/splatforge/mesh/native/simplify.h','src/splatforge/mesh/native/triangulate.h','src/splatforge/mesh/native/expansion.h','src/splatforge/mesh/native/patch.h'],
        extra_compile_args=['/fp:strict','/utf-8'] if sys.platform=='win32' else ['-fno-fast-math','-ffp-contract=off'],
    )]
setup(ext_modules=extensions)
