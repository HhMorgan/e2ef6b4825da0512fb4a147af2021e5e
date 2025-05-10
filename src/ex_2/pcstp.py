import argparse
import os

import gurobipy as gp
import matplotlib.pyplot as plt
import networkx as nx
from gurobipy import GRB

from src.util.utils import rooted_proper_subsets


def read_instance_file(filename: str | os.PathLike) -> nx.Graph:
    with open(filename, "r", encoding="utf-8") as f:
        n_nodes = int(f.readline())
        m_edges = int(f.readline())

        G = nx.Graph()

        for _ in range(n_nodes):
            line = f.readline()
            node_id, prize = line.split()
            G.add_node(int(node_id), prize=int(prize))

        for _ in range(m_edges):
            line = f.readline()
            values = line.split()
            if len(values) == 4:
                G.add_edge(int(values[1]), int(values[2]),
                           id=int(values[0]),
                           cost=int(values[3]))

        return G


def get_selected_edge_ids(model: gp.Model, graph: nx.Graph) -> list[int]:
    selected_edges: list[int] = []
    if model.SolCount > 0:
        for i, j, data in graph.edges(data=True):
            edge_id = int(data['id'])
            x_e = model.getVarByName(f'x[{edge_id}]')

            if x_e.X >= 1:
                print(f"Added edge [{i},{j}] with cost {data['cost']}")
                selected_edges.append(edge_id)

    return selected_edges


def build_model(model: gp.Model, graph: nx.Graph):
    graph.undirected = True  # make sure the graph is undirected

    node_indices = [n for n in graph]  # grab the ID of every node in the graph
    root = 0
    edge_data = [(data['id'], i, j) for i, j, data in graph.edges(data=True)]

    reversed_arcs = {(j, i) for (i, j) in graph.edges}
    arcs = reversed_arcs.union(set(graph.edges))  # edges in graph and their inverted counterpart
    arcs_with_zero = arcs.union((root, j) for j in node_indices)

    # create a directed NX graph from all the arcs (useful for querying incident arcs later)
    digraph = nx.DiGraph()
    digraph.add_edges_from(arcs)

    digraph_with_zero = digraph.copy()
    digraph_with_zero.add_node(0)
    digraph_with_zero.add_edges_from((0, j) for j in node_indices)

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
    for subset in rooted_proper_subsets(node_indices + [root]):
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

    graph = read_instance_file(args.filename)

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
