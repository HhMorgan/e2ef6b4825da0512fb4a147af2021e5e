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
    n, m = processing_times.shape

    p = list(processing_times)
    pie = list(machine_sequences)

    x = model.addVars(
        [(i,j,k) for i in range(m) for j in range(n) for k in range(n)],
        name="x",
        vtype=GRB.BINARY,
    )

    s = model.addVars(
        [(i,j) for i in range(m) for j in range(n)],
        name="s",
        vtype=GRB.INTEGER,
    )

    # constraints
    C = n * m * processing_times.max()
    model.addConstrs((s[i, j] >= s[i, k] + p[k][i] - C * x[i, j, k] for i in range(m) for j in range(n) for k in range(n) if j != k),
                     name="A job j can only start after its preceding job k finished")
    #model.addConstrs((s[i,k] + p[k][i] <= s[i,j] + C * x[i,j,k] for i in range(m) for j in range(n) for k in range(n) if j != k),
    #                 name="A job j can only start after its preceding job k finished")
    model.addConstrs((s[i,j] + p[j][i] <= s[i,k] + C * (1 - x[i,j,k]) for i in range(m) for j in range(n) for k in range(n) if j != k),
                     name="A job j must start before its succeeding job k is started")

    model.addConstrs((gp.quicksum(x[i,j,k] for j in range(n) for k in range(n) if j != k) * 2 == n*(n-1) for i in range(m)),
                     name="On every machine, the sum of job precedence must be n(n-1)/2")

    model.addConstrs((s[pie[j][l],j] >= s[pie[j][l-1],j] + p[j][pie[j][l-1]] for l in range(1, m) for j in range(n)),
                     name="Enforce the correct order of machines for job j")


    # we want to get the sum of the job completion times of the last job on the last machine
    model.setObjective(gp.quicksum(s[pie[j][m-1],j] + p[j][pie[j][m-1]] for j in range(n)), GRB.MINIMIZE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", default="../../instances/ex1.2-instance.dat")
    #parser.add_argument("--filename", default="../../instances/ex1.2-instance_small.dat")
    args = parser.parse_args()

    processing_times, machine_sequences = read_instance_file(args.filename)
    # n_jobs, n_machines = processing_times.shape

    model = gp.Model("ex1.2")
    build_model(model, processing_times, machine_sequences)

    model.update()
    model.optimize()

    if model.status == GRB.INF_OR_UNBD or model.status == GRB.INFEASIBLE:
        print("Model is infeasible. Computing IIS...")
        model.computeIIS()
        model.write("model.ilp")

    if model.SolCount > 0:
        print(f"obj. value = {model.ObjVal}")
        for v in model.getVars():
            print(f"{v.VarName} = {v.X}")

    model.close()
