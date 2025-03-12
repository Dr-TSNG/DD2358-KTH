# KTH DD2358 High Performance Computing final project

This repository contains the final project for the course DD2358 High Performance Computing at KTH.

## Project structure

```bash
.
├── solver              # The solver
│   ├── cy.pyx          # Cython implementation of the solver
│   ├── gpu.py          # GPU implementation of the solver
│   ├── native.py       # Native Python implementation of the solver(baseline)
│   └── setup.py
├── example.py          # Example usage of the solver
├── img                 # Images used in the report
├── oldcode             # Old code that was used for profiling
└── report.typ          # report source file
```

## Description

This project aims at improving the performance of [Spectral Solver for Navier-Stokes Equations](https://github.com/pmocz/navier-stokes-spectral-python). The original code is written in Python and uses NumPy for the computations.