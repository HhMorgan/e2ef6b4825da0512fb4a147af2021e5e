import argparse
import os
from pathlib import Path

import gurobipy as gp
import networkx as nx
from gurobipy import GRB

from src.util.utils import generate_names, generate_name_matrix


def read_instance_file(filename: str | os.PathLike) -> nx.Graph:
    with open(Path(filename), mode="r", encoding="utf-8") as f:
        n_nodes = int(f.readline())
        n_edges = int(f.readline())

        graph = nx.Graph()

        # skip comment line
        f.readline()

        # read node lines
        for _ in range(n_nodes):
            line = f.readline()
            node_id, name, supply_demand = line.split()
            graph.add_node(int(node_id), name=name, supply_demand=int(supply_demand))

        # skip comment line
        f.readline()

        # read edge lines
        for _ in range(n_edges):
            line = f.readline()
            (
                edge_id,
                node_1,
                node_2,
                transport_cost,
                build_cost_1,
                build_cost_2,
                capacity_1,
                capacity_2,
            ) = line.split()
            graph.add_edge(
                int(node_1),
                int(node_2),
                id=int(edge_id),
                transport_cost=int(transport_cost),
                build_cost_1=int(build_cost_1),
                build_cost_2=int(build_cost_2),
                capacity_1=int(capacity_1),
                capacity_2=int(capacity_2),
            )

        return graph


def build_model(model: gp.Model, graph: nx.Graph):
    n = len(graph.nodes)
    m = len(graph.edges)

    demands = generate_names(n, "b")

    transportation_costs = generate_name_matrix(n, m, "c")
    building_cost_1 = generate_name_matrix(n, m, "d1")
    building_cost_1 = generate_name_matrix(n, m, "d2")
    capacity_1 = generate_name_matrix(n, m, "u1")
    capacity_2 = generate_name_matrix(n, m, "u2")

    flows = generate_name_matrix(n, m, "f")
    is_1_built = generate_name_matrix(n, m, "x1")
    is_2_built = generate_name_matrix(n, m, "x2")

    # note that nodes are 1-indexed

    # put your model building code here
    #
    # x = model.addVars(...)

    #
    # if you want to access your variables outside this function, you can use
    # model._x = x
    # to save a reference in the model itself
    #
    # model.addConstrs(...)


    model.setObjective(GRB.MINIMIZE)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", default="instances/ex1.1-instance.dat")
    args = parser.parse_args()

    graph = read_instance_file(args.filename)

    model = gp.Model("ex1.1")
    build_model(model, graph)

    model.update()
    model.optimize()

    if model.SolCount > 0:
        print(f"obj. value = {model.ObjVal}")
        for v in model.getVars():
            print(f"{v.VarName} = {v.X}")

    model.close()
