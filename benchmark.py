import time
import matplotlib.pyplot as plt
import numpy as np
from solver.native import native_solver
from solver.cy import cython_solver
from solver.torch import torch_solver

t = 0  # current time of the simulation
tEnd = 1  # time at which simulation ends
dt = 0.001  # timestep
nu = 0.001  # viscosity

def test_solver_performance(solver, scales, runs, warmup=False,**kwargs):
    results = []
    print(f"Running {solver.__name__} for scales {scales} and {runs} runs each")
    for scale in scales:
        run_times = []
        print(f"Running {solver.__name__} for scale {scale}")
        if warmup:
            solver(scale, t, tEnd, dt, nu, **kwargs)
        for _ in range(runs):
            start_time = time.time()
            solver(scale, t, tEnd, dt, nu, **kwargs)
            elapsed_time = time.time() - start_time
            run_times.append(elapsed_time)
        avg_time = np.mean(run_times)
        stddev_time = np.std(run_times)
        results.append((scale, avg_time, stddev_time))
    print(f"Results for {solver.__name__}:")
    for scale, avg_time, stddev_time in results:
        print(f"Scale {scale}: {avg_time:.2f} ± {stddev_time:.2f} seconds")
    return np.array(results)

def main():
    #scales = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    #runs = 10
    scales = [100, 200, 300, 400, 500]
    runs = 3

    results_native = test_solver_performance(native_solver, scales, runs)
    results_cython = test_solver_performance(cython_solver, scales, runs)
    results_torch = test_solver_performance(torch_solver, scales, runs, device="cuda")

    # save results data
    np.save("results_native.npy", results_native)
    np.save("results_cython.npy", results_cython)
    np.save("results_torch.npy", results_torch)

    plt.figure()
    # use log scale for y-axis
    plt.yscale("symlog")
    plt.errorbar(results_native[:, 0], results_native[:, 1], yerr=results_native[:, 2], label="native")
    plt.errorbar(results_cython[:, 0], results_cython[:, 1], yerr=results_cython[:, 2], label="cython")
    plt.errorbar(results_torch[:, 0], results_torch[:, 1], yerr=results_torch[:, 2], label="torch")

    plt.xlabel("Spatial resolution")
    plt.ylabel("Average runtime (s)")
    plt.legend()
    plt.grid()
    plt.savefig("benchmark.png", dpi=240)
    plt.show()

if __name__ == "__main__":
    main()



