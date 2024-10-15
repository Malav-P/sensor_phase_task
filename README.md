## Overview
This repository contains code for the paper \<insert paper here once published\> on the concurrent optimization of satellite phasing and tasking for cislunar SDA. See the paper for details on formulation. In the `src` directory, you can find a `main.py` file which contains a function `run_experiment`. This is the main call in most experiments.


## Installation
After cloning the repository to your machine, create a conda environment
```bash
conda env create -f environment.yml
```

Note that a gurobi license is required to run the experiments in this repository.

## Experiments
All experiments are under the `experiments/` directory as jupyter notebooks.