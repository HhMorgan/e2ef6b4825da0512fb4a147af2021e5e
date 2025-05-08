# Programming Project - Part 1

_Hesham Morgan, Johannes Riedmann_

## Requirements

- Python 3.12 installed
- Env-variable `GRB_LICENSE_FILE` set to the path of a Gurobi licence file

## Setup

The framework uses [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage the Python project and its dependencies.

Run `uv sync` in the project's root directory to set up the virtual environment with all dependencies.
```sh
uv sync
```

## Deployment

### Running a single optimization

To run the optimization of a specific formulation use the following command in the project's root directory

```sh
uv run src/project/part_1/kmst.py -k 4 --instance "instances/project/g01.dat" --formulation seq
```

Replace the k-value, the instance file, and the respective formulation (`"seq", "scf", "mcf"`) as you please.

### Running the benchmarking

To do some benchmarking on increasingly large instances, run the `jobs.py` script instead.

**Method 1:** Test all formulations (seq, scf, mcf) at the same time:
```sh
uv run src/project/part_1/jobs.py
```
This will save various statistics from the optimization runs to a CSV file called
`results.csv` that is created in the project's root directory.

**Method 2:** Test only a specific formulation (seq, scf, mcf), add the `--formulation` option:
```sh
uv run src/project/part_1/jobs.py --formulation seq
```
This creates a file called `results_[formulation].csv` to store the benchmark statistics.