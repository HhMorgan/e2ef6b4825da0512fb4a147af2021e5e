import argparse
import csv
import datetime
import os
import sys
from pathlib import Path
from xmlrpc.client import MAXINT

import gurobipy as gp
import matplotlib.pyplot as plt
import networkx as nx
from gurobipy import GRB
from gurobipy._statusconst import StatusConstClass as SC

from cuts import lazy_constraint_callback
from model import create_model, get_selected_edge_ids
from util import read_instance, write_solution


def log(message: str) -> None:
    current_time = datetime.datetime.now().time()
    print(f"[{current_time.isoformat(timespec="seconds")}]: {message}", flush=True)


def execute_lp(args):
    instance_name = Path(args.instance).stem
    model_name = f"{instance_name}_{args.k}_{args.formulation}"

    print("-" * 40)
    log(f"Reading instance {instance_name} for model [{model_name}]...")
    G: nx.Graph = read_instance(args.instance)
    G.undirected = True  # make sure the graph is undirected

    # guard clauses for nonsensical inputs
    if args.k < 0:
        sys.exit(f"Cannot return a subgraph with a negative number of vertices! Received k = {args.k}")
    elif args.k == 0:
        sys.exit("Trivial solution: The empty set is a minimal subtree of G with 0 vertices.")
    elif args.k == 1:
        sys.exit("Trivial solution: Take any vertex of G for a minimum subtree with just 1 vertex.")

    # context handlers take care of disposing resources correctly
    with gp.Model(model_name) as model:
        model._original_graph = G
        model._k = args.k
        model._formulation = args.formulation

        # some parameters to control Gurobi's output and other aspects in the solution process
        # feel free to change them / add new ones as you see fit
        # (see https://docs.gurobi.com/projects/optimizer/en/current/concepts/parameters.html)
        # model.Params.MIPFocus = 2
        # model.Params.OutputFlag = 0   # hide verbose gurobi output
        model.Params.LogToConsole = 0  # hide verbose gurobi output
        model.Params.LogFile = "gurobi.log"

        log(f"Building model [{model_name}]...")
        create_model(model, G, args.k)
        model.update()

        if not model.IsMIP:
            sys.exit(f"Error: Your formulation for '{args.formulation}' is not a (mixed) integer linear program.")
        if model.IsQP or model.IsQCP:
            sys.exit(f"Error: Your formulation for '{args.formulation}' is non-linear.")

        # write model to file in readable format (useful for debugging)
        # model.write("model.lp")

        # set thread, time and memory limit
        if args.threads:
            model.Params.Threads = args.threads
        if args.timelimit:
            model.Params.TimeLimit = args.timelimit
        if args.memory_limit:
            model.Params.SoftMemLimit = args.memory_limit

        # tell Gurobi that the model is not complete for CEC and DCC formulations (needs to be considered in presolving)
        if args.formulation in {"cec", "dcc"}:
            model.Params.LazyConstraints = 1

        try:
            log(f"Optimizing model [{model_name}]...")
            if args.formulation in {"cec", "dcc"}:
                model.optimize(lazy_constraint_callback)
            else:
                model.optimize()
        except gp.GurobiError as e:
            log(f"ERROR in optimization: " + str(e))

        model.printStats()

        # create dict from gurobi status codes
        # taken from: https://support.gurobi.com/hc/en-us/community/posts/360047967872/comments/360012141411
        status_names = {SC.__dict__[k]: k for k in SC.__dict__.keys() if 'A' <= k[0] <= 'Z'}

        if model.Status == GRB.Status.OPTIMAL or model.Status == GRB.Status.SUBOPTIMAL:
            log(f"Finished optimization of model [{model_name}]!")

            # check solution feasibility
            selected_edges = set(get_selected_edge_ids(model, G))
            k_mst = G.edge_subgraph(edge for edge in G.edges if G.edges[edge]["id"] in selected_edges)

            if k_mst.number_of_nodes() == 0:
                sys.exit("Error: Received an empty subgraph.")
            if not nx.is_tree(k_mst):
                print("Error: the provided solution is not a tree!")
                print(f"{k_mst.number_of_nodes()=}")
                print(f"{k_mst.number_of_edges()=}")
                print(f"{nx.is_tree(k_mst)=}")
                print(f"{nx.number_connected_components(k_mst)=}")
            else:
                print("k-MST is valid")

            print(f"Objective value: {model.ObjVal}")

            # draw graph for debugging purposes
            if args.show:
                nx.draw_kamada_kawai(k_mst, with_labels=True)
                plt.show()
            # draw base graph for debugging purposes
            # if args.showOG:
            #     nx.draw(G, with_labels=True)
            #     plt.show()
            #
            # if args.printArcs:
            #     chosen_edges = list(k_mst.edges())
            #     chosen_edges.sort()
            #     for (i, j) in chosen_edges:
            #         print(f"({i},{j})")

        else:
            log(f"Optimization aborted [{status_names[model.Status]}]")

        if args.results_file:
            # collect statistics
            results = {
                "name": args.run_name,
                "instance": Path(args.instance).stem,
                "k": args.k,
                "formulation": args.formulation.upper(),
                "status": status_names[model.Status],
                # beware: Rounding infinity throws an error
                "objVal": model.ObjVal if model.ObjVal > MAXINT else round(model.ObjVal),
                "bestBound": model.ObjBound if model.ObjBound > MAXINT else round(model.ObjBound, 1),
                "gap": round(model.MIPGap, 4),
                "runtime": f"{min(model.runtime, model.Params.TimeLimit):.3f}",
                "n": round(model.NodeCount)
            }

            with open(args.results_file, "a", encoding="utf-8", newline='') as f:
                # if results file is empty, start with the CSV header
                if os.stat(args.results_file).st_size == 0:
                    header = ",".join(results.keys())
                    f.write(header + "\n")
                # write the benchmarking results to the given output file in CSV format
                writer = csv.writer(f)
                writer.writerow(results.values())

        if args.solution_file:
            write_solution(args.solution_file, get_selected_edge_ids(model, G))


if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser(description="ILP-based k-MST solver")
    parser.add_argument("--run-name", type=str, required=False, help="title of the current run")
    parser.add_argument("--instance", type=str, required=True, help="path to instance file")
    parser.add_argument("-k", type=int, required=True, help="instance parameter k")
    parser.add_argument("--formulation", required=True, choices=["seq", "scf", "mcf", "cec", "dcc"])
    parser.add_argument("--results-file", type=str, help="path to results file")
    parser.add_argument("--solution-file", type=str, help="path to solution file")
    parser.add_argument("--threads", type=int, default=1, help="maximum number of threads to use")
    parser.add_argument("--timelimit", type=int, default=3600, help="time limit (in seconds)")
    parser.add_argument("--memory-limit", type=float, default=8, help="memory limit (in GB)")
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, help="draw graph in a debug view")
    parser.add_argument("--showOG", action=argparse.BooleanOptionalAction, help="draw graph in a debug view")
    parser.add_argument("--printArcs", action=argparse.BooleanOptionalAction, help="draw graph in a debug view")
    args = parser.parse_args()

    execute_lp(args)
