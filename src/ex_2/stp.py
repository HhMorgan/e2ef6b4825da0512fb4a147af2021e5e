import argparse
import os

import gurobipy as gp
import matplotlib.pyplot as plt
import networkx as nx
from gurobipy import GRB
from networkx.algorithms.boundary import edge_boundary


def subsets(iterable):
    total_length = len(iterable)
    masks = [1 << i for i in range(total_length)]

    for i in range(1, (1 << total_length) - 2):
        yield [subset for mask, subset in zip(masks, iterable) if i & mask]


def read_instance_file(filename: str | os.PathLike) -> nx.Graph:
    with open(filename, "r", encoding="utf-8") as f:
        n_nodes = int(f.readline())
        m_edges = int(f.readline())

        G = nx.Graph()

        for _ in range(n_nodes):
            line = f.readline()
            node_id, vertex_set = line.split()
            G.add_node(int(node_id), terminal=vertex_set == 'T')

        for _ in range(m_edges):
            line = f.readline()
            values = line.split()
            if len(values) == 4:
                G.add_edge(int(values[1]), int(values[2]),
                           id=int(values[0]),
                           weight=int(values[3]))

        return G

def get_selected_edge_ids(model: gp.Model, graph: nx.Graph) -> list[int]:
    selected_edges: list[int] = []
    if model.SolCount > 0:
        for i,j,data in graph.edges(data=True):
            edge_id = int(data['id'])
            x_e = model.getVarByName(f'x[{edge_id}]')

            if x_e.X >= 1:
                print(edge_id)
                selected_edges.append(edge_id)

    return selected_edges


def build_model(model: gp.Model, graph: nx.Graph):
    graph.undirected = True  # make sure the graph is undirected

    node_indices = [n for n in graph]  # grab the ID of every node in the graph
    edge_data = [(data['id'], i,j) for i,j,data in graph.edges(data=True)]

    reversed_arcs = {(j, i) for (i, j) in graph.edges}
    arcs = reversed_arcs.union(set(graph.edges))  # edges in graph and their inverted counterpart
    arcs_with_zero = arcs.union((0, j) for j in node_indices)

    # create a directed NX graph from all the arcs (useful for querying incident arcs later)
    digraph = nx.DiGraph()
    digraph.add_edges_from(arcs)

    digraph_with_zero = digraph.copy()
    digraph_with_zero.add_node(0)
    digraph_with_zero.add_edges_from((0, j) for j in node_indices)

    terminal_vertices = list([t for t in graph.nodes if graph.nodes[t]['terminal']])
    root = terminal_vertices[0]

    # variables
    x = model.addVars(
        [e for e,i,j in edge_data],
        name="x",
        lb=0,
        # ub=1,
        vtype=GRB.CONTINUOUS
    )
    y = model.addVars(
        [(i, j) for (i, j) in arcs],
        name="y",
        vtype=GRB.BINARY
    )

    # constraints
    for subset in subsets(terminal_vertices):
        if root in subset:
            edges_out_of_p = edge_boundary(digraph, subset)
            edges_into_p = ((j, i) for (i, j) in edges_out_of_p)
            model.addConstr(gp.quicksum(y[i, j] for (i, j) in edges_into_p) >= 1, "P subsets")

    model.addConstr(gp.quicksum(y[i, j] for (i, j) in arcs if j in terminal_vertices) == len(terminal_vertices) - 1,
                    name="form a tree")

    model.addConstrs((x[e] == y[i, j] + y[j, i]
                      for e,i,j in edge_data), "edge inclusion")

    # objective function
    model.setObjective(gp.quicksum(x[e] * graph.edges[i,j]["weight"] for e,i,j in edge_data),
                       GRB.MINIMIZE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", default="instances/ex2/stp-instance.dat")
    args = parser.parse_args()

    graph = read_instance_file(args.filename)

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
