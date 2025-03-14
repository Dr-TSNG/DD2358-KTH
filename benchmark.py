import time
import matplotlib.pyplot as plt
import numpy as np
from solver.naive import naive_solver
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
    scales = [100, 200, 300, 400, 500, 600, 700, 800]
    runs = 10

    results_torch = test_solver_performance(torch_solver, scales, runs, warmup=True, device="cuda")
    np.save("results_torch.npy", results_torch)
    
    results_cython = test_solver_performance(cython_solver, scales, runs)
    np.save("results_cython.npy", results_cython)

    results_naive = test_solver_performance(naive_solver, scales, runs)
    np.save("results_naive.npy", results_naive)

    plt.figure()
    # use log scale for y-axis
    plt.yscale("symlog")
    plt.errorbar(results_naive[:, 0], results_naive[:, 1], yerr=results_naive[:, 2], label="naive")
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



