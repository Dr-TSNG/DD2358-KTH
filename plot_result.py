import numpy as np
import matplotlib.pyplot as plt

def main():
    results_naive = np.load("results_naive.npy")
    results_cython = np.load("results_cython.npy")
    results_torch = np.load("results_torch.npy")

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
    plt.title("Solver Performance Comparison: Naive vs Cython vs Torch")
    plt.savefig("result.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()