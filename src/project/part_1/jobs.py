import math
import os
import sys
import argparse
from pathlib import Path
from kmst import execute_lp

class GurobiArguments:
    instance: str = ""
    k: int = 1
    formulation: str = ""
    memory_limit: int = 8
    threads: int = 1
    timelimit: int = 3600
    show: bool = False
    results_file: str = ""
    solution_file: str = ""
    run_name: str = ""

    def __init__(self, memory_limit: int, threads: int, timeout_seconds: int):
        self.memory_limit = memory_limit
        self.threads = threads
        self.timelimit = timeout_seconds


def run_jobs(instance_dir: str, results_file: str, *, specific_formulation: str = None):
    k_factors = [0.2, 0.5]

    if specific_formulation:
        formulations = [specific_formulation]
    else:
        formulations = ['seq', 'scf', 'mcf']

    args = GurobiArguments(8, 1, 3600)
    args.show = False
    args.results_file = results_file

    for filename in sorted(os.listdir(instance_dir)):
        filepath = os.path.join(instance_dir, filename)

        args.instance = filepath

        if os.path.isfile(filepath):
            # determine total number of vertices in instance
            with open(filepath, 'r') as instance:
                first_line = instance.readline()
                vertex_count = int(first_line)

            # loop over k values
            for factor in k_factors:
                args.k = math.ceil(factor * vertex_count)
                # loop over formulations
                for formulation in formulations:
                    args.formulation = formulation
                    args.run_name = f'{Path(filepath).stem}:{factor}:{formulation}'

                    execute_lp(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmarking procedure of ILP formulations for a k-MST")
    parser.add_argument("--instances", type=str, default="./instances/project", help="path to instance directory")
    parser.add_argument("--results", type=str, default="./results.csv", help="path to results file")
    parser.add_argument("--formulation", type=str, choices=["seq", "scf", "mcf", "cec", "dcc"], help="choose a specific formulation")
    args = parser.parse_args()

    # if a specific formulation was chosen, append that formulation to the results file name
    if args.formulation:
        args.results = args.results.replace(".csv", f"_{args.formulation}.csv")

    # quick checks of the given paths
    if not os.path.exists(args.instances):
        sys.exit(f"Could not find instances directory '{os.path.abspath(args.instances)}'.")
    if not os.path.isdir(args.instances):
        sys.exit(f"Instances path is not a directory! Path: '{os.path.abspath(args.instances)}'")

    if os.path.isdir(args.results):
        sys.exit("Given path for results file refers to a directory!")

    open(args.results, 'w').close() # clear current file contents

    run_jobs(args.instances, args.results, specific_formulation=args.formulation)