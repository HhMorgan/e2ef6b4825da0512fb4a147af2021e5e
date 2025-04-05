# Mathematical Programming, Exercise Sheet 1 Python Framework

This framework is provided to help you get started with implementing the models on the first exercise sheet. Feel free to alter the code as you see fit.

It uses Python 3.12 and Gurobi 12 (via `gurobipy`).

## Using `uv`

The framework uses `uv` to manage the Python project and its dependencies.

### Installing `uv`

See https://docs.astral.sh/uv/getting-started/installation/ on how to install it.

### Setting up the virtual enrivonment

Run `uv sync` to set up the virtual environment with all dependencies.

### Adding dependencies

Run `uv add name_of_python_package` to add a new dependency to `pyproject.toml`.


## Installing Gurobi

The Gurobi version included with `gurobipy` supports models with up to 2k variables and constraints, which should be enough for this exercise's instances.

For larger instances (like you'll find in the programming project), you'll need a license. You can get a free academic named-user license to run Gurobi on your computer if you register with your TU-Wien-provided email address. See here for details: https://www.gurobi.com/features/academic-named-user-license/


## Using the framework

Run

```
uv run src/mathprog_ex1/ex1.1.py
uv run src/mathprog_ex1/ex1.2.py
uv run src/mathprog_ex1/ex1.3.py
```

from the project root directory (the one containing `pyproject.toml`) to run the code for the respective exercise.