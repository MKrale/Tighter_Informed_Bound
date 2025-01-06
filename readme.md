# Tighter POMDP Bounds Repository

Repository containing code, as well as gathered data, as used for the following submission

> *Redacted* 
> Tighter Value-Function Approximations for POMDPs

This code is currently structured to make it easy to test and verify the results shown in this paper. After publication, we will also publish the code for computing our bounds as a seperate Julia module so that it can easily be used by others.

## Contents

This repository contains the following files:

Folders:

  - **TIB**                 : Contains all code related to our novel bounds TIB, ETIB and OTIB (here denoted as TIB, ETIB and OTIB). The most important files within are:
    - **solver.jl**         : Contains the algorithms for computing the bounds, implemented using the *POMDPs.jl* framework.
    - **SimpleHeuristics.jl**: Contains a custom implementation of FIB and QMDP.
    - **Caching.jl**        : Contains code for precomputing beliefs and probabilities.
  - **Sarsop_altered**      : Contains a copy of the NativeSARSOP solver, with alterations as explained in the paper.
  - **Data**                : Contains all data used for our experiments
  - **Environments**        : Contains all benchmarks in *POMDPs.jl* format that are not publically available elsewhere. The most imporant files within are:
    - **Sparse_models**      : Contains all environments collected form pomdps.org

Relevant scripts:

  - **RunAll.sh**             : Automatically runs all experiments used in the paper.
  - **run\*.jl**              : Scripts for running single experiments.
  - **plotting_python.ipynb** : Notebook used for data collection & plotting.


## Getting started

After cloning this repository, run the followig commands to install all packages:
```bash
julia --project=.
]
instantiate
```

## How to run

To run all experiments from the paper at once, run the following:
```bash
bash ./Runall.sh
```
To run a single test, run either ```run_upperbound.jl``` or ```run_sarsoptest.jl``` with flag ```-h``` to see the available options.