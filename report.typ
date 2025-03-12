= Level D - bottlenecks analysis

#figure(
  grid(
    image("img/cprofile_og_1.png"),
    image("img/cprofile_og_2.png"),
  ),
  caption: [cProfile for the unoptimized code],
)

From the cProfile output, we can see that the majority of time is spent in FFT functions (`fftn, ifftn`), especially within `grad` and `diffusion_solve` functions.

= Level C - Optimization with compiled code

== Use pyFFTW library

Since FFT operations are the bottlenecks, our primary goal is to optimize `ffftn` and `ifftn` to make the code run faster. We can optimize this part of the code by using the FFTW library, which is a highly optimized FFT library.

There are two steps to optimize the FFT operations:

+ Replace NumPy FFT with pyFFTW: Utilize optimized FFT implementations.

+ Precompute FFT Plans: Reduce overhead by reusing FFT configurations.

This can be achieved by following changes in the code:

```python
import pyfftw
from pyfftw.interfaces.numpy_fft import fftn, ifftn, ifftshift

pyfftw.interfaces.cache.enable()
pyfftw.config.NUM_THREADS = 4

// Remove all np.fft prefixes to use the pyFFTW variant
```

== Compile the code with Cython

To further optimize the code, we can compile the Python code using Cython. Cython is a static compiler that allows us to write C extensions for Python.

// todo: Add Cython code snippet

== Results

After optimizing the code with pyFFTW and Cython, we can see a significant improvement in the performance of the code. The optimized code runs much faster than the unoptimized version.

#figure(
  image("img/cprofile_pyfftw.png"),
  caption: [cProfile for pyFFTW optimized code],
)

#figure(
  image("img/cprofile_pyfftw_cython.png"),
  caption: [cProfile for pyFFTW and Cython optimized code],
)
