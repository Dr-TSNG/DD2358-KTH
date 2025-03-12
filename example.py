import matplotlib.pyplot as plt
from solver.native import native_solver

"""
Simulate the Navier-Stokes equations (incompressible viscous fluid) 
with a Spectral method

Original code by Philip Mocz (2023), @PMocz
https://github.com/pmocz/navier-stokes-spectral-python
"""


def main():
    N = 400  # Spatial resolution
    t = 0  # current time of the simulation
    tEnd = 1  # time at which simulation ends
    dt = 0.001  # timestep
    tOut = 0.01  # draw frequency
    nu = 0.001  # viscosity

    wzs = native_solver(N, t, tEnd, dt, tOut, nu)

    for i in range(1, len(wzs) * tOut):
        # clac the actual index
        index = int(i / tOut)
        plt.cla()
        plt.imshow(wzs[index], cmap="RdBu")
        plt.clim(-20, 20)
        ax = plt.gca()
        ax.invert_yaxis()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_aspect("equal")
        plt.pause(0.001)

    plt.savefig("navier-stokes-spectral.png", dpi=240)
    plt.show()
