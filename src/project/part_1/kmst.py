import argparse
import gurobipy as gp
import json
from pathlib import Path
import networkx as nx
import sys

from model import create_model, lazy_constraint_callback, get_selected_edge_ids
from util import read_instance, write_solution

if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser(description="ILP-based k-MST solver")
    parser.add_argument("--instance", type=str, required=True, help="path to instance file")
    parser.add_argument("--k", type=int, required=True, help="instance parameter k")
    parser.add_argument("--formulation", required=True, choices=["seq", "scf", "mcf", "cec", "dcc"])
    parser.add_argument("--results-file", type=str, help="path to results file")
    parser.add_argument("--solution-file", type=str, help="path to solution file")
    parser.add_argument("--threads", type=int, default=1, help="maximum number of threads to use")
    parser.add_argument("--timelimit", type=int, default=3600, help="time limit (in seconds)")
    parser.add_argument("--memorylimit", type=float, default=8, help="memory limit (in GB)")
    args = parser.parse_args()


    inst = Path(args.instance).stem
    model_name = f"{inst}_{args.k}_{args.formulation}"

    G: nx.Graph = read_instance(args.instance)
    # hint: use a directed graph in your formulations! add an artificial root node!

    # context handlers take care of disposing resources correctly
    with gp.Model(model_name) as model:
        model._original_graph = G
        model._k = args.k
        model._formulation = args.formulation

        create_model(model)
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
        if args.memorylimit:
            model.Params.SoftMemLimit = args.memorylimit

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
        selected_edges = set(get_selected_edge_ids(model))
        k_mst = G.edge_subgraph(edge for edge in G.edges if G.edges[edge]["id"] in selected_edges)
        if not nx.is_tree(k_mst):
            print("Error: the provided solution is not a tree!")
            print(f"{k_mst.number_of_nodes()=}")
            print(f"{k_mst.number_of_edges()=}")
            print(f"{nx.is_tree(k_mst)=}")
            print(f"{nx.number_connected_components(k_mst)=}")
        else:
            print("k-MST is valid")
            

        # print statistics
        results = {
            "instance": args.instance,
            "k": args.k,
            "formulation": args.formulation,
            "status": model.Status,
            "objective_value": model.ObjVal,
            "best_bound": model.ObjBound,
            "gap": round(model.MIPGap, 4),
            "runtime": round(model.runtime, 3),
            "n_nodes": round(model.NodeCount)
        }
        print(results)
        if args.results_file:
            with open(args.results_file, "w", encoding="utf-8") as f:
                json.dump(results, f)

        if args.solution_file:
            write_solution(args.solution_file, get_selected_edge_ids(model))
            pass