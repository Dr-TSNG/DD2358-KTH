from solver.naive import naive_solver
import cProfile

def main():
    N = 400  # Spatial resolution
    t = 0  # current time of the simulation
    tEnd = 1  # time at which simulation ends
    dt = 0.001  # timestep
    nu = 0.001  # viscosity
    
    pr = cProfile.Profile()
    pr.enable()
    naive_solver(N, t, tEnd, dt, nu)
    pr.disable()
    pr.print_stats(sort="cumulative")
    pr.dump_stats("naive.prof")

if __name__ == "__main__":
    main()
