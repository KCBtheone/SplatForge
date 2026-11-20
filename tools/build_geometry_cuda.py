"""Compile the owned conservative filter to portable PTX, without Torch."""
from pathlib import Path
import shutil
import subprocess

source = Path(__file__).resolve().parents[1]/'src/splatforge/mesh/native/separate.cu'
compiler = shutil.which('nvcc')
if compiler is None:
    raise SystemExit('nvcc is required only to rebuild the distributed PTX')
subprocess.run([compiler, '--ptx', '-arch=compute_60', '--fmad=false', '-O3',
                str(source), '-o', str(source.with_suffix('.ptx'))], check=True)
