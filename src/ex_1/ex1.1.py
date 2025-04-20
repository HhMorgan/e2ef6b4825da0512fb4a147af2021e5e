import argparse
import os
from pathlib import Path

from src.util.utils import generate_latex_table, generate_three_tables_per_page,generate_three_long_tables_per_page


import gurobipy as gp
import networkx as nx
from gurobipy import GRB


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
    graph.undirected = True  # make sure the graph is undirected

    node_indices = [n for n in graph] # grab the ID of every node in the graph

    reversed_edges = {(j, i) for (i, j) in graph.edges}
    present_edges = reversed_edges.union(set(graph.edges))  # edges in graph and their inverted counterpart

    # complete_graph = {(i, j) for i in node_indices for j in node_indices if i != j}  # set comprehension
    # edge_complement = complete_graph.difference(present_edges)

    # variables
    f = model.addVars(
        [(i, j) for (i, j) in present_edges],
        name="f",
        vtype=GRB.CONTINUOUS
    )
    x1 = model.addVars(
        [(i, j) for (i, j) in present_edges],
        name="x1",
        vtype=GRB.BINARY
    )
    x2 = model.addVars(
        [(i, j) for (i, j) in present_edges],
        name="x2",
        vtype=GRB.BINARY
    )

    # constraints
    model.addConstrs((f[i, j] >= 0 for (i, j) in present_edges), "positive flow")

    for i in node_indices:
        sum_incoming = gp.quicksum(f[u, v] for (u, v) in present_edges if v == i)
        sum_outgoing = gp.quicksum(f[v, u] for (v, u) in present_edges if v == i)
        model.addConstr(sum_outgoing - sum_incoming == graph.nodes[i]['supply_demand'], name="flow_conservation")

    for (i, j) in present_edges: # add constraints for both directions of each edge
        props = graph.edges[i, j]
        model.addConstr(x1[i, j] + x2[i, j] <= 1, name=f"only_build_one")
        model.addConstr(f[i, j] <= x1[i, j] * props['capacity_1'] + x2[i, j] * props['capacity_2'],
                        name=f"constrained_flow")

    # objective function
    objective_func = gp.quicksum(
        f[i, j] * graph.edges[i, j]["transport_cost"] +
        x1[i, j] * graph.edges[i, j]["build_cost_1"] +
        x2[i, j] * graph.edges[i, j]["build_cost_2"]
        for (i, j) in present_edges)

    model.setObjective(objective_func, GRB.MINIMIZE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", default="../../instances/ex1/ex1.1-instance.dat")
    args = parser.parse_args()

    graph = read_instance_file(args.filename)

    model = gp.Model("ex1.1")
    build_model(model, graph)

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
    latex_table = generate_three_long_tables_per_page(model)
    model.close()
