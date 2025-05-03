import argparse
import gurobipy as gp
import csv
from pathlib import Path
import networkx as nx
import sys
import matplotlib.pyplot as plt

from model import create_model, lazy_constraint_callback, get_selected_edge_ids
from util import read_instance, write_solution


def execute_lp(args):
    inst = Path(args.instance).stem
    model_name = f"{inst}_{args.k}_{args.formulation}"

    G: nx.Graph = read_instance(args.instance)

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

        # some parameters to control Gurobi's output and other aspects in the solution process
        # feel free to change them / add new ones as you see fit
        # (see https://docs.gurobi.com/projects/optimizer/en/current/concepts/parameters.html)
        # model.Params.OutputFlag = 0
        # model.Params.MIPFocus = 2

        if args.formulation in {"cec", "dcc"}:
            model.optimize(lazy_constraint_callback)
        else:
            model.optimize()

        model.printStats()


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

        if args.show:
            nx.draw(k_mst, with_labels=True)
            plt.show()

        # print statistics
        results = {
            "run": args.run_name,
            "instance": Path(args.instance).stem,
            "k": args.k,
            "formulation": args.formulation,
            "status": model.Status,
            "objective_value": model.ObjVal,
            "best_bound": model.ObjBound,
            "gap": round(model.MIPGap, 4),
            "runtime": round(model.runtime, 3),
            "n_nodes": round(model.NodeCount)
        }

        if args.results_file:
            # write the benchmarking results to the given output file in CSV format
            with open(args.results_file, "a", encoding="utf-8", newline='') as f:
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
    args = parser.parse_args()

    execute_lp(args)