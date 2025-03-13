"""
Pytorch Solver for Navier-Stokes Spectral Method Simulation
"""

import torch
import torch.fft

def __poisson_solve(rho, kSq_inv):
    """solve the Poisson equation, given source field rho on GPU"""
    V_hat = -(torch.fft.fftn(rho)) * kSq_inv
    V = torch.real(torch.fft.ifftn(V_hat))
    return V


def __diffusion_solve(v, dt, nu, kSq):
    """solve the diffusion equation over a timestep dt, given viscosity nu on GPU"""
    v_hat = torch.fft.fftn(v) / (1.0 + dt * nu * kSq)
    v = torch.real(torch.fft.ifftn(v_hat))
    return v


def __grad(v, kx, ky):
    """return gradient of v on GPU"""
    v_hat = torch.fft.fftn(v)
    dvx = torch.real(torch.fft.ifftn(1j * kx * v_hat))
    dvy = torch.real(torch.fft.ifftn(1j * ky * v_hat))
    return dvx, dvy


def __div(vx, vy, kx, ky):
    """return divergence of (vx,vy) on GPU"""
    dvx_x = torch.real(torch.fft.ifftn(1j * kx * torch.fft.fftn(vx)))
    dvy_y = torch.real(torch.fft.ifftn(1j * ky * torch.fft.fftn(vy)))
    return dvx_x + dvy_y


def __curl(vx, vy, kx, ky):
    """return curl of (vx,vy) on GPU"""
    dvx_y = torch.real(torch.fft.ifftn(1j * ky * torch.fft.fftn(vx)))
    dvy_x = torch.real(torch.fft.ifftn(1j * kx * torch.fft.fftn(vy)))
    return dvy_x - dvx_y


def __apply_dealias(f, dealias):
    """apply 2/3 rule dealias to field f on GPU"""
    f_hat = dealias * torch.fft.fftn(f)
    return torch.real(torch.fft.ifftn(f_hat))


def __iter_loop(vx, vy, kx, ky, kSq, kSq_inv, dealias, t, Nt, dt, nu):
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

        # Vorticity (for plotting)
        wz_series.append(__curl(vx, vy, kx, ky))

        t += dt

    return torch.stack(wz_series).cpu().numpy()


def torch_solver(N, t, tEnd, dt, nu, device="cuda"):
    """Solver for Navier-Stokes equation optimized for GPU"""

    # Domain [0,1] x [0,1]
    L = 1
    xlin = torch.linspace(0, L, steps=N+1, device=device, dtype=torch.float64)
    xlin = xlin[:N]  # chop off the periodic point
    xx, yy = torch.meshgrid(xlin, xlin, indexing='xy')

    # Initial condition (vortex)
    vx = -torch.sin(2 * torch.pi * yy).to(device)
    vy = torch.sin(2 * torch.pi * xx * 2).to(device)

    # Fourier Space Variables
    klin = (2.0 * torch.pi / L) * torch.arange(-N / 2, N / 2, device=device)
    kmax = torch.max(klin)
    kx, ky = torch.meshgrid(klin, klin, indexing='xy')
    kx = torch.fft.ifftshift(kx)
    ky = torch.fft.ifftshift(ky)
    kSq = kx ** 2 + ky ** 2
    kSq_inv = 1.0 / kSq
    kSq_inv[kSq == 0] = 1

    # Dealias with the 2/3 rule
    dealias = ((torch.abs(kx) < (2.0 / 3.0) * kmax) & (torch.abs(ky) < (2.0 / 3.0) * kmax)).to(device)

    # Number of timesteps
    Nt = int(torch.ceil(torch.tensor(tEnd / dt)))

    return __iter_loop(vx, vy, kx, ky, kSq, kSq_inv, dealias, t, Nt, dt, nu)