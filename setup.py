import numpy as np
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("cylibrary.pyx"),
    include_dirs=[np.get_include()],
    install_requires=['numpy', 'pyfftw'],
)
