# KTH DD2358 High Performance Computing final project

This repository contains the final project for the course DD2358 High Performance Computing at KTH.

## Project structure

```bash
.
├── solver              # The solver
│   ├── cy.pyx          # Cython implementation of the solver
│   ├── gpu.py          # GPU implementation of the solver
│   └── naive.py        # naive Python implementation of the solver(baseline)
│   └── __init__.py
├── example.py          # Example usage of the solver
├── prof.py             # Profiling script
├── setup.py            # Setup file for building the Cython module
```

## Description

This project aims at improving the performance of [Spectral Solver for Navier-Stokes Equations](https://github.com/pmocz/navier-stokes-spectral-python). The original code is written in Python and uses NumPy for the computations.

## Usage

This project requires NumPy, Matplotlib, SciPy, PyTorch, and Cython.

To build the Cython module, run:

```bash
python setup.py build_ext --inplace
```

To run the example, execute:

```bash
python example.py
```
