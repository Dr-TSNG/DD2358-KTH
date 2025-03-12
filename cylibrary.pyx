import numpy as np
cimport numpy as cnp
cimport cython
import pyfftw
from pyfftw.interfaces.numpy_fft import fftn, ifftn

# Enable FFTW optimizations
pyfftw.interfaces.cache.enable()
pyfftw.config.NUM_THREADS = 8


@cython.boundscheck(False)
@cython.wraparound(False)
def poisson_solve(cnp.ndarray[cnp.double_t, ndim=2] rho,
                 cnp.ndarray[cnp.double_t, ndim=2] kSq_inv):
    cdef cnp.ndarray[cnp.complex128_t, ndim=2] rho_hat = fftn(rho)
    cdef cnp.ndarray[cnp.complex128_t, ndim=2] V_hat = -rho_hat * kSq_inv
    return np.real(ifftn(V_hat))


@cython.boundscheck(False)
@cython.wraparound(False)
def diffusion_solve(cnp.ndarray[cnp.double_t, ndim=2] v,
                   cnp.float64_t dt,
                   cnp.float64_t nu,
                   cnp.ndarray[cnp.double_t, ndim=2] kSq):
    cdef cnp.ndarray[cnp.complex128_t, ndim=2] v_hat = fftn(v)
    v_hat /= (1.0 + dt*nu*kSq)
    return np.real(ifftn(v_hat))


@cython.boundscheck(False)
@cython.wraparound(False)
def grad(cnp.ndarray[cnp.double_t, ndim=2] v,
         cnp.ndarray[cnp.double_t, ndim=2] kx,
         cnp.ndarray[cnp.double_t, ndim=2] ky):
    cdef cnp.ndarray[cnp.complex128_t, ndim=2] v_hat = fftn(v)
    cdef cnp.ndarray[cnp.complex128_t, ndim=2] term_x = 1j * kx * v_hat
    cdef cnp.ndarray[cnp.complex128_t, ndim=2] term_y = 1j * ky * v_hat
    return np.real(ifftn(term_x)), np.real(ifftn(term_y))


@cython.boundscheck(False)
@cython.wraparound(False)
def div(cnp.ndarray[cnp.double_t, ndim=2] vx,
        cnp.ndarray[cnp.double_t, ndim=2] vy,
        cnp.ndarray[cnp.double_t, ndim=2] kx,
        cnp.ndarray[cnp.double_t, ndim=2] ky):
    cdef cnp.ndarray[cnp.complex128_t, ndim=2] vx_hat = fftn(vx)
    cdef cnp.ndarray[cnp.complex128_t, ndim=2] vy_hat = fftn(vy)
    cdef cnp.ndarray[cnp.complex128_t, ndim=2] term_x = 1j * kx * vx_hat
    cdef cnp.ndarray[cnp.complex128_t, ndim=2] term_y = 1j * ky * vy_hat
    return np.real(ifftn(term_x + term_y))


@cython.boundscheck(False)
@cython.wraparound(False)
def curl(cnp.ndarray[cnp.double_t, ndim=2] vx,
         cnp.ndarray[cnp.double_t, ndim=2] vy,
         cnp.ndarray[cnp.double_t, ndim=2] kx,
         cnp.ndarray[cnp.double_t, ndim=2] ky):
    cdef cnp.ndarray[cnp.complex128_t, ndim=2] vx_hat = fftn(vx)
    cdef cnp.ndarray[cnp.complex128_t, ndim=2] vy_hat = fftn(vy)
    cdef cnp.ndarray[cnp.complex128_t, ndim=2] term_x = 1j * ky * vx_hat
    cdef cnp.ndarray[cnp.complex128_t, ndim=2] term_y = -1j * kx * vy_hat
    return np.real(ifftn(term_x - term_y))


@cython.boundscheck(False)
@cython.wraparound(False)
def apply_dealias(cnp.ndarray[cnp.double_t, ndim=2] f,
                 cnp.ndarray[cnp.uint8_t, ndim=2] dealias):
    cdef cnp.ndarray[cnp.complex128_t, ndim=2] f_hat = fftn(f)
    f_hat *= dealias
    return np.real(ifftn(f_hat))
