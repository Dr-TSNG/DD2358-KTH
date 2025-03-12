import numpy as np

"""
Native Python Solver for Navier-Stokes Spectral Method Simulation
Original code by Philip Mocz (2023), @PMocz
https://github.com/pmocz/navier-stokes-spectral-python
"""

def __poisson_solve(rho, kSq_inv):
    """solve the Poisson equation, given source field rho"""
    V_hat = -(np.fft.fftn(rho)) * kSq_inv
    V = np.real(np.fft.ifftn(V_hat))
    return V


def __diffusion_solve(v, dt, nu, kSq):
    """solve the diffusion equation over a timestep dt, given viscosity nu"""
    v_hat = (np.fft.fftn(v)) / (1.0 + dt * nu * kSq)
    v = np.real(np.fft.ifftn(v_hat))
    return v


def __grad(v, kx, ky):
    """return gradient of v"""
    v_hat = np.fft.fftn(v)
    dvx = np.real(np.fft.ifftn(1j * kx * v_hat))
    dvy = np.real(np.fft.ifftn(1j * ky * v_hat))
    return dvx, dvy


def __div(vx, vy, kx, ky):
    """return divergence of (vx,vy)"""
    dvx_x = np.real(np.fft.ifftn(1j * kx * np.fft.fftn(vx)))
    dvy_y = np.real(np.fft.ifftn(1j * ky * np.fft.fftn(vy)))
    return dvx_x + dvy_y


def __curl(vx, vy, kx, ky):
    """return curl of (vx,vy)"""
    dvx_y = np.real(np.fft.ifftn(1j * ky * np.fft.fftn(vx)))
    dvy_x = np.real(np.fft.ifftn(1j * kx * np.fft.fftn(vy)))
    return dvy_x - dvx_y


def __apply_dealias(f, dealias):
    """apply 2/3 rule dealias to field f"""
    f_hat = dealias * np.fft.fftn(f)
    return np.real(np.fft.ifftn(f_hat))

def __native_loop(vx, vy, kx, ky, kSq, kSq_inv, dealias, t, Nt, dt, nu):
    wz_series = []

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

        # vorticity (for plotting)
        wz_series.append(__curl(vx, vy, kx, ky))
        t += dt

    return wz_series



def native_solver(N, t, tEnd, dt, nu):

    # N         = 400     # Spatial resolution
    # t         = 0       # current time of the simulation
    # tEnd      = 1       # time at which simulation ends
    # dt        = 0.001   # timestep
    # tOut      = 0.01    # draw frequency
    # nu        = 0.001   # viscosity
	
    # Domain [0,1] x [0,1]
    L = 1    
    xlin = np.linspace(0,L, num=N+1)  # Note: x=0 & x=1 are the same point!
    xlin = xlin[0:N]                  # chop off periodic point
    xx, yy = np.meshgrid(xlin, xlin)

    # Intial Condition (vortex)
    vx = -np.sin(2*np.pi*yy)
    vy =  np.sin(2*np.pi*xx*2) 
	
	# Fourier Space Variables
    klin = 2.0 * np.pi / L * np.arange(-N/2, N/2)
    kmax = np.max(klin)
    kx, ky = np.meshgrid(klin, klin)
    kx = np.fft.ifftshift(kx)
    ky = np.fft.ifftshift(ky)
    kSq = kx**2 + ky**2
    with np.errstate(divide='ignore'):
        kSq_inv = np.divide(1.0, kSq)
    kSq_inv[kSq==0] = 1

	# dealias with the 2/3 rule
    dealias = (np.abs(kx) < (2./3.)*kmax) & (np.abs(ky) < (2./3.)*kmax)

    # number of timesteps
    Nt = int(np.ceil(tEnd/dt))
    return __native_loop(vx, vy, kx, ky, kSq, kSq_inv, dealias, t, Nt, dt, nu)
