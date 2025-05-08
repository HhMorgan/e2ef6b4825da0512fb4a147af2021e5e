import argparse
import os
from pathlib import Path

import gurobipy as gp
import networkx as nx
from gurobipy import GRB


def read_instance_file(filename: str | os.PathLike) -> nx.Graph:
    with open(filename, "r", encoding="utf-8") as f:
        n_nodes = int(f.readline())
        n_edges = int(f.readline())

        G = nx.Graph()
        G.add_nodes_from(range(1, n_nodes + 1))

        for line in f:
            values = [int(i) for i in line.split()]
            if len(values) == 4:
                G.add_edge(values[1], values[2], id=values[0], weight=values[3])

        return G


def build_model(model: gp.Model, graph: nx.Graph):
    graph.undirected = True  # make sure the graph is undirected

    node_indices = [n for n in graph]  # grab the ID of every node in the graph

    reversed_arcs = {(j, i) for (i, j) in graph.edges}
    arcs = reversed_arcs.union(set(graph.edges))  # edges in graph and their inverted counterpart
    arcs_with_zero = arcs.union((0, j) for j in node_indices)

    # create a directed NX graph from all the arcs (useful for querying incident arcs later)
    digraph = nx.DiGraph()
    digraph.add_edges_from(arcs)

    digraph_with_zero = digraph.copy()
    digraph_with_zero.add_node(0)
    digraph_with_zero.add_edges_from((0, j) for j in node_indices)


    # variables
    x = model.addVars(
        [e for e in graph.edges],
        name="x",
        lb=0,
        vtype=GRB.CONTINUOUS
    )
    y = model.addVars(
        [(i, j) for (i, j) in arcs],
        name="y",
        vtype=GRB.BINARY
    )
    v = model.addVars(
        [i for i in node_indices],
        name="v",
        vtype=GRB.BINARY
    )

    # constraints
    # TODO fix this
    model.addConstrs((gp.quicksum(y[i,j] for (i,j) in arcs) >= 1
                      for i in node_indices), "P subsets")

    model.addConstr(gp.quicksum(y[0,j] for j in node_indices) == 1, name="one root edge")

    model.addConstrs((x[(i,j)] == y[i,j] + y[j,i]
                      for (i,j) in graph.edges), "edge inclusion")

    # TODO Does not work and would be a non-linear constraint
    model.addConstrs((gp.quicksum(y[i, j] * v[j]) == gp.quicksum(v[i])
                    for (i,j) in arcs_with_zero), name="one root edge")

    # TODO Impossible constraint - we don't know V' yet
    v_prime = 10
    model.addConstr(gp.quicksum(v[i] for i in node_indices) == v_prime, name="idk")

    # objective function
    model.setObjective(gp.quicksum(v[i] * graph.nodes[i]["price"] for i in graph) -
                       gp.quicksum(x[e] * graph.edges[e]["weight"] for e in graph.edges),
                       GRB.MAXIMIZE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", default="../../instances/ex2/pcstp-instance.dat")
    args = parser.parse_args()

    graph = read_instance_file(args.filename)

    model = gp.Model("Price-collecting Steiner Tree Problem")
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

    model.close()
