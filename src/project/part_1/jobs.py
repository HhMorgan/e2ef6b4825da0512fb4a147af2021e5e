import math
import os
from kmst import execute_lp

class Arguments:
    instance = ""
    k = 1
    formulation = ""
    memorylimit = 8
    threads = 1
    timelimit = 3600
    show = False
    results_file = ""
    solution_file = ""
    run_name = ""


def run_jobs(instance_dir: str, results_file: str):
    k_factors = [0.2, 0.5]
    formulations = ['seq', 'scf', 'mcf']

    args = Arguments()
    args.memorylimit = 8
    args.threads = 1
    args.timelimit = 3600
    args.show = False
    args.results_file = results_file
    args.solution_file = ""

    for filename in os.listdir(instance_dir):
        filepath = os.path.join(instance_dir, filename)

        args.instance = filepath

        if os.path.isfile(filepath):
            with open(filepath, 'r') as instance:
                first_line = instance.readline()
                num_vertices = int(first_line)

                for factor in k_factors:
                    args.k = math.ceil(factor * num_vertices)

                    for formulation in formulations:
                        args.formulation = formulation
                        args.run_name = f'{filename}_{factor}_{formulation}'

                        execute_lp(args)


if __name__ == '__main__':
    instance_directory = "instances/project"
    results_file = "results.csv"

    with open(results_file, "w", encoding="utf-8", newline='') as f:
        header = "run,instance,k,formulation,status,objective_value,best_bound,gap,runtime,n_nodes"
        f.write(header + "\n")

    run_jobs(instance_directory, results_file)