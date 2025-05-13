import argparse

import gurobipy as gp
import matplotlib.pyplot as plt
import networkx as nx
from gurobipy import GRB
from networkx.algorithms.boundary import edge_boundary

from src.util.utils import rooted_proper_subsets, read_instance_file, get_selected_edge_ids


def build_model(model: gp.Model, graph: nx.Graph):
    graph.undirected = True  # make sure the graph is undirected

    node_indices = [n for n in graph]  # grab the ID of every node in the graph
    edge_data = [(data['id'], i, j) for i, j, data in graph.edges(data=True)]

    reversed_arcs = {(j, i) for (i, j) in graph.edges}
    arcs = reversed_arcs.union(set(graph.edges))  # edges in graph and their inverted counterpart

    # create a directed NX graph from all the arcs (useful for querying incident arcs later)
    digraph = graph.to_directed()

    terminal_vertices = list([t for t in graph.nodes if graph.nodes[t]['terminal']])
    root = terminal_vertices[0]
    terminal_vertices = set(terminal_vertices)

    # variables
    x = model.addVars(
        [e for e, i, j in edge_data],
        name="x",
        lb=0,
        vtype=GRB.CONTINUOUS
    )
    y = model.addVars(
        [(i, j) for (i, j) in arcs],
        name="y",
        vtype=GRB.BINARY
    )

    # print(f"root = {root}")
    # constraints
    for subset in rooted_proper_subsets(node_indices, root):
        if not len(terminal_vertices.difference(set(subset))) == 0:
            edges_out_of_p = edge_boundary(digraph, subset)
            model.addConstr(gp.quicksum(y[i, j] for (i, j) in edges_out_of_p) >= 1, "P_subsets")

    model.addConstr(gp.quicksum(y[i, j] for (i, j) in arcs if j in terminal_vertices) == len(terminal_vertices) - 1,
                    name="form_a_tree")

    model.addConstrs((x[e] == y[i, j] + y[j, i]
                      for e, i, j in edge_data), "bind_x_to_y")

    # objective function
    model.setObjective(gp.quicksum(x[e] * graph.edges[i, j]["weight"] for e, i, j in edge_data),
                       GRB.MINIMIZE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", default="instances/ex2/stp-instance_small.dat")
    args = parser.parse_args()

    # describe how the instance file should be parsed
    node_params = {
        "terminal": lambda values: (values[1] == 'T'),
    }
    edge_params = {
        "id": lambda values: int(values[0]),
        "weight": lambda values: int(values[3]),
    }
    graph = read_instance_file(args.filename, node_params, edge_params)

    model = gp.Model("Steiner Tree Problem")
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

        nx.draw(k_mst, with_labels=True)
        plt.show()

    model.close()
