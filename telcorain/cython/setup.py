from __future__ import annotations

from pathlib import Path
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PYX = HERE / "raincolor.pyx"

ext = Extension(
    name="telcorain.cython.raincolor",
    sources=[str(PYX)],
    include_dirs=[np.get_include()],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
)

setup(
    name="telcorain",
    ext_modules=cythonize(
        [ext],
        compiler_directives={"language_level": "3"},
    ),
)
