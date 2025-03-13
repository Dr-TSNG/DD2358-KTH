import numpy as np
import matplotlib.pyplot as plt

def main():
    results_native = np.load("results_native.npy")
    results_cython = np.load("results_cython.npy")
    results_torch = np.load("results_torch.npy")

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