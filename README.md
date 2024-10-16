## Overview
This repository contains code for the paper \<insert paper here once published\> on the concurrent optimization of satellite phasing and tasking for cislunar SDA. See the paper for details on formulation. In the `src` directory, you can find a `main.py` file which contains a function `run_experiment`. This is the main call in most experiments. Gravity dynamics and propagators are in the `data_util/` folder.


## Installation
After cloning the repository to your machine, create a conda environment
```bash
conda env create -f environment.yml
```

### Gurobi license
The installation of gurobipy comes with a restricted license. If you have an unrestricted license file elsewhere on your machine, you must replace the restricted license with your license. To find the location of the your restricted license, run the following with the conda environment (or virtual environment) activated:

```bash
(sensortask)username@usernames-MBP ~ % gurobi_cl --license
```

You should see an output showing the path to the file

```bash
Using license at /path/to/restricted/license/dir/gurobi.lic
```

Remove this license

```bash
rm /path/to/restricted/license/dir/gurobi.lic
```

and copy over the unrestricted license to this location

```bash
cp /path/to/unrestricted/license/dir/gurobi.lic /path/to/restricted/license/dir/
```

Note that you may be able to find the unrestricted license path by running the `gurobi_cl --license`command with all virtual enviroments deactivated.

Note that an unrestricted gurobi license is required to run the experiments in this repository.

## Experiments
All experiments are under the `experiments/` directory as jupyter notebooks.