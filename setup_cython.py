from setuptools import setup
from Cython.Build import cythonize
import numpy as np

modules = cythonize('sampler.pyx')
for m in modules:
    m.extra_compile_args.append('-O3')
setup(
    ext_modules=cythonize("sampler.pyx"),
    include_dirs=np.get_include()
)