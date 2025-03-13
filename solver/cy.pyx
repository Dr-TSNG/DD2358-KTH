"""
Cython Solver for Navier-Stokes Spectral Method Simulation
"""

import numpy as np
cimport numpy as cnp
cimport cython
import pyfftw
from pyfftw.interfaces.numpy_fft import fftn, ifftn, ifftshift




@cython.boundscheck(False)
@cython.wraparound(False)
def __poisson_solve(cnp.ndarray[cnp.double_t, ndim=2] rho,
                 cnp.ndarray[cnp.double_t, ndim=2] kSq_inv):
    cdef cnp.ndarray[cnp.complex128_t, ndim=2] rho_hat = fftn(rho)
    cdef cnp.ndarray[cnp.complex128_t, ndim=2] V_hat = -rho_hat * kSq_inv
    return np.real(ifftn(V_hat))


@cython.boundscheck(False)
@cython.wraparound(False)
def __diffusion_solve(cnp.ndarray[cnp.double_t, ndim=2] v,
                   cnp.float64_t dt,
                   cnp.float64_t nu,
                   cnp.ndarray[cnp.double_t, ndim=2] kSq):
    cdef cnp.ndarray[cnp.complex128_t, ndim=2] v_hat = fftn(v)
    v_hat /= (1.0 + dt*nu*kSq)
    return np.real(ifftn(v_hat))


@cython.boundscheck(False)
@cython.wraparound(False)
def __grad(cnp.ndarray[cnp.double_t, ndim=2] v,
         cnp.ndarray[cnp.double_t, ndim=2] kx,
         cnp.ndarray[cnp.double_t, ndim=2] ky):
    cdef cnp.ndarray[cnp.complex128_t, ndim=2] v_hat = fftn(v)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] dvx = np.real(ifftn(1j * kx * v_hat))
    cdef cnp.ndarray[cnp.float64_t, ndim=2] dvy = np.real(ifftn(1j * ky * v_hat))
    return dvx, dvy


@cython.boundscheck(False)
@cython.wraparound(False)
def __div(cnp.ndarray[cnp.double_t, ndim=2] vx,
        cnp.ndarray[cnp.double_t, ndim=2] vy,
        cnp.ndarray[cnp.double_t, ndim=2] kx,
        cnp.ndarray[cnp.double_t, ndim=2] ky):
    cdef cnp.ndarray[cnp.complex128_t, ndim=2] vx_hat = fftn(vx)
    cdef cnp.ndarray[cnp.complex128_t, ndim=2] vy_hat = fftn(vy)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] dvx_x = np.real(ifftn(1j * kx * vx_hat))
    cdef cnp.ndarray[cnp.float64_t, ndim=2] dvy_y = np.real(ifftn(1j * ky * vy_hat))
    return dvx_x + dvy_y


@cython.boundscheck(False)
@cython.wraparound(False)
def __curl(cnp.ndarray[cnp.double_t, ndim=2] vx,
         cnp.ndarray[cnp.double_t, ndim=2] vy,
         cnp.ndarray[cnp.double_t, ndim=2] kx,
         cnp.ndarray[cnp.double_t, ndim=2] ky):
    cdef cnp.ndarray[cnp.complex128_t, ndim=2] vx_hat = fftn(vx)
    cdef cnp.ndarray[cnp.complex128_t, ndim=2] vy_hat = fftn(vy)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] dvx_y = np.real(ifftn(1j * ky * vx_hat))
    cdef cnp.ndarray[cnp.float64_t, ndim=2] dvy_x = np.real(ifftn(1j * kx * vy_hat))
    return dvy_x - dvx_y


@cython.boundscheck(False)
@cython.wraparound(False)
def __apply_dealias(cnp.ndarray[cnp.double_t, ndim=2] f,
                 cnp.ndarray[cnp.uint8_t, ndim=2] dealias):
    cdef cnp.ndarray[cnp.complex128_t, ndim=2] f_hat = dealias * fftn(f)
    return np.real(ifftn(f_hat))

@cython.boundscheck(False)
@cython.wraparound(False)
cdef list __native_loop(cnp.ndarray[cnp.double_t, ndim=2] vx, cnp.ndarray[double, ndim=2] vy, 
                        cnp.ndarray[cnp.double_t, ndim=2] kx, cnp.ndarray[double, ndim=2] ky, 
                        cnp.ndarray[cnp.double_t, ndim=2] kSq, cnp.ndarray[double, ndim=2] kSq_inv, 
                        cnp.ndarray[cnp.uint8_t, ndim=2] dealias, double t, int Nt, double dt, double nu):
    cdef list wz_series = []
    cdef int i
    cdef cnp.ndarray[double, ndim=2] dvx_x, dvx_y, dvy_x, dvy_y, rhs_x, rhs_y, div_rhs, P, dPx, dPy
    for i in range(Nt):
        dvx_x, dvx_y = __grad(vx, kx, ky)
        dvy_x, dvy_y = __grad(vy, kx, ky)

        rhs_x = -(vx * dvx_x + vy * dvx_y)
        rhs_y = -(vx * dvy_x + vy * dvy_y)

        rhs_x = __apply_dealias(rhs_x, dealias)
        rhs_y = __apply_dealias(rhs_y, dealias)

        vx += dt * rhs_x
        vy += dt * rhs_y

        # Poisson solve for pressure
        div_rhs = __div(rhs_x, rhs_y, kx, ky)
        P = __poisson_solve(div_rhs, kSq_inv)
        dPx, dPy = __grad(P, kx, ky)

        # Correction (to eliminate divergence component of velocity)
        vx += -dt * dPx
        vy += -dt * dPy

        # Diffusion solve (implicit)
        vx = __diffusion_solve(vx, dt, nu, kSq)
        vy = __diffusion_solve(vy, dt, nu, kSq)

        # Vorticity (for plotting)
        wz_series.append(__curl(vx, vy, kx, ky))
        t += dt

    return wz_series

def cython_solver(int N, double t, double tEnd, double dt, double nu, int n_threads=4):
    """
    Python wrapper function for the solver.
    """
    # Enable FFTW optimizations
    pyfftw.interfaces.cache.enable()
    # Set number of threads, use nproc
    pyfftw.config.NUM_THREADS = n_threads
    L = 1
    xlin = np.linspace(0, L, num=N+1)  
    xlin = xlin[0:N]  # Chop off periodic point
    xx, yy = np.meshgrid(xlin, xlin)

    # Initial Condition (vortex)
    vx = -np.sin(2 * np.pi * yy)
    vy = np.sin(2 * np.pi * xx * 2)

    # Fourier Space Variables
    klin = 2.0 * np.pi / L * np.arange(-N/2, N/2)
    kmax = np.max(klin)
    kx, ky = np.meshgrid(klin, klin)
    kx = ifftshift(kx)
    ky = ifftshift(ky)
    kSq = kx**2 + ky**2
    with np.errstate(divide='ignore'):
        kSq_inv = np.divide(1.0, kSq)
    kSq_inv[kSq == 0] = 1

    # Dealias with the 2/3 rule
    dealias = (np.abs(kx) < (2. / 3.) * kmax) & (np.abs(ky) < (2. / 3.) * kmax).astype(np.uint8)
    
    # Number of timesteps
    Nt = int(np.ceil(tEnd / dt))
    return __native_loop(vx, vy, kx, ky, kSq, kSq_inv, dealias, t, Nt, dt, nu)
