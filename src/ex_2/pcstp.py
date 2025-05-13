import argparse

import gurobipy as gp
import matplotlib.pyplot as plt
import networkx as nx
from gurobipy import GRB

from src.util.utils import rooted_proper_subsets, read_instance_file, get_selected_edge_ids


def build_model(model: gp.Model, graph: nx.Graph):
    graph.undirected = True  # make sure the graph is undirected

    node_indices = [n for n in graph]  # grab the ID of every node in the graph
    root = 0
    edge_data = [(data['id'], i, j) for i, j, data in graph.edges(data=True)]

    reversed_arcs = {(j, i) for (i, j) in graph.edges}
    arcs = reversed_arcs.union(set(graph.edges))  # edges in graph and their inverted counterpart
    arcs_with_zero = arcs.union((root, j) for j in node_indices)

    # create a directed NX graph from all the arcs (useful for querying incident arcs later)
    digraph = graph.to_directed()

    digraph_with_zero = digraph.copy()
    digraph_with_zero.add_node(0)
    digraph_with_zero.add_edges_from([(0, j) for j in node_indices], cost=0)

    # variables
    x = model.addVars(
        [e for e, i, j in edge_data],
        name="x",
        lb=0,
        vtype=GRB.CONTINUOUS
    )
    y = model.addVars(
        [(i, j) for (i, j) in arcs_with_zero],
        name="y",
        vtype=GRB.BINARY
    )
    v = model.addVars(
        node_indices,
        name="v",
        vtype=GRB.BINARY
    )

    # constraints
    for subset in rooted_proper_subsets(node_indices + [root], root):
        edges_out_of_s = nx.edge_boundary(digraph_with_zero, subset)
        model.addConstr(gp.quicksum(y[i, j] for (i, j) in edges_out_of_s) >= 1, "S_subsets")

    model.addConstr(gp.quicksum(y[0, j] for j in node_indices) == 1, name="one_root_edge")

    model.addConstrs((x[e] == y[i, j] + y[j, i]
                      for e, i, j in edge_data), "edge_inclusion")

    model.addConstrs((y[i, j] <= v[j]
                      for (i, j) in arcs_with_zero), "vertex_inclusion")

    # objective function
    model.setObjective(gp.quicksum(v[i] * graph.nodes[i]["prize"] for i in graph) -
                       gp.quicksum(x[data["id"]] * data["cost"] for u, v, data in graph.edges(data=True)),
                       GRB.MAXIMIZE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", default="instances/ex2/pcstp-instance_medium.dat")
    args = parser.parse_args()

    # describe how the instance file should be parsed
    node_params = {
        "prize": lambda values: int(values[1]),
    }
    edge_params = {
        "id": lambda values: int(values[0]),
        "cost": lambda values: int(values[3]),
    }
    graph = read_instance_file(args.filename, node_params, edge_params)

    model = gp.Model("Prize-collecting Steiner Tree Problem")
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

        selected_edges = get_selected_edge_ids(model, graph)
        k_mst = graph.edge_subgraph(e for e in graph.edges if graph.edges[e]["id"] in selected_edges)

        nx.draw_kamada_kawai(k_mst, with_labels=True)
        plt.show()

    model.close()
