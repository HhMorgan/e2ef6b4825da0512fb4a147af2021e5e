import argparse
import os
from pathlib import Path

import gurobipy as gp
import numpy as np
from gurobipy import GRB


def read_instance_file(filename: str | os.PathLike) -> tuple[np.ndarray, np.ndarray]:
    with open(Path(filename), mode="r", encoding="utf-8") as f:
        n_jobs = int(f.readline())
        n_machines = int(f.readline())

        # skip comment line
        f.readline()

        proc_times = []
        for _ in range(n_jobs):
            proc_times_j = [int(p) for p in f.readline().split()]
            assert len(proc_times_j) == n_machines
            proc_times.append(proc_times_j)
        processing_times = np.array(proc_times, dtype=np.int32)

        # skip comment line
        f.readline()
        machine_seq = []
        for _ in range(n_jobs):
            machine_seq_j = [int(h) for h in f.readline().split()]
            assert set(machine_seq_j) == set(range(n_machines))
            machine_seq.append(machine_seq_j)
        machine_sequences = np.array(machine_seq, dtype=np.int32)

        return processing_times, machine_sequences


def build_model(model: gp.Model, processing_times: np.ndarray, machine_sequences: np.ndarray):
    # note that both jobs and machines are 0-indexed here
    n_jobs, n_machines = processing_times.shape

    # put your model building code here
    #
    # x = model.addVars(...)
    #
    # if you want to access your variables outside this function, you can use
    # model._x = x
    # to save a reference in the model itself
    #
    # model.addConstrs(...)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", default="instances/ex1.2-instance.dat")
    args = parser.parse_args()

    processing_times, machine_sequences = read_instance_file(args.filename)
    n_jobs, n_machines = processing_times.shape

    model = gp.Model("ex1.2")
    build_model(model, processing_times, machine_sequences)

    model.update()
    model.optimize()

    if model.SolCount > 0:
        print(f"obj. value = {model.ObjVal}")
        for v in model.getVars():
            print(f"{v.VarName} = {v.X}")

    model.close()
