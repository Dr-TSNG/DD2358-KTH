import numpy as np
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        name="solver.cy",
        sources=["solver/cy.pyx"],
    ),
]

setup(
    ext_modules=cythonize(extensions),
    include_dirs=[np.get_include()],
    install_requires=['numpy', 'scipy'],
)
