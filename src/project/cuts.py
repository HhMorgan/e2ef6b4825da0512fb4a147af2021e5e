import gurobipy as gp
import networkx as nx
from gurobipy import GRB

EPSILON: float = 1e-05


def lazy_constraint_callback(model: gp.Model, where):
    new_cuts: int = 0

    # callback was invoked because the solver found an optimal integral solution
    if where == GRB.Callback.MIPSOL:
        # check integer solution for feasibility
        if model._formulation == "cec":
            new_cuts = find_violated_cec_int(model)
        elif model._formulation == "dcc":
            new_cuts = find_violated_dcc_int(model)

    # increment the counter for the total number of added cutting planes
    model._cuts += new_cuts


def find_violated_cec_int(model: gp.Model) -> int:
    # Create a graph including all edges who's x-value is (roughly) 1
    x = model._x
    G = nx.Graph()
    for (i, j), x_var in x.items():
        val = model.cbGetSolution(x_var)
        if val > 1 - EPSILON:
            G.add_edge(i, j)

    # Detect cycles
    try:
        C = nx.find_cycle(G)
        # Add lazy constraint to eliminate this cycle
        model.cbLazy(gp.quicksum(x[i, j] + x[j, i] for i, j in C) <= len(C) - 1)
        return 1
    except nx.NetworkXNoCycle:
        return 0


def find_violated_dcc_int(model: gp.Model) -> int:
    # add your DCC separation code here
    digraph_with_zero = model._digraph_with_zero.copy()
    digraph = model._graph.copy()
    x = model._x
    y = model._y
    source_vertex = 0
    cuts_added = 0

    # Label all arcs with weight w_ij = x_ij
    for i, j in digraph_with_zero.edges():
        x_var = model.cbGetSolution(x[i, j])
        digraph_with_zero[i][j]['weight'] = min(1.0, max(x_var, 0.0))

    for target_node in digraph.nodes():
        y_value = model.cbGetSolution(y[target_node])
        if y_value <= EPSILON:
            continue

        cut_val, (s, t) = nx.minimum_cut(digraph_with_zero, _s=source_vertex, _t=target_node, capacity='weight')
        if cut_val < y_value - EPSILON and target_node in t:
            model.cbLazy(
                gp.quicksum(x[u, v] for u, v in digraph_with_zero.edges() if u in s and v in t)
                >= y[target_node]
            )
            cuts_added += 1

    return cuts_added