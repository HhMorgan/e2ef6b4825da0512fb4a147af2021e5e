import gurobipy as gp
import networkx as nx
from gurobipy import GRB

EPSILON: float = 1e-05

def lazy_constraint_callback(model: gp.Model, where):
    # callback was invoked because the solver found an optimal integral solution
    if where == GRB.Callback.MIPSOL:
        # check integer solution for feasibility
        if model._formulation == "cec":
            find_violated_cec_int(model)
        elif model._formulation == "dcc":
            find_violated_dcc_int(model)


    # callback was invoked because the solver found an optimal but fractional solution
    elif where == GRB.Callback.MIPNODE and model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
        # check fractional solution to find violated CECs/DCCs to strengthen the bound
        if model._formulation == "cec":
            find_violated_cec_float(model)
        elif model._formulation == "dcc":
            find_violated_dcc_float(model)


def find_violated_cec_int(model: gp.Model):
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
    except nx.NetworkXNoCycle:
        return


def find_violated_dcc_int(model: gp.Model):
    # add your DCC separation code here
    digraph_with_zero = model._digraph_with_zero.copy()
    digraph = model._graph.copy()
    x = model._x
    y = model._y
    source_vertex = 0

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


def find_violated_cec_float(model: gp.Model):
    # Check how deep we are in exploring LP nodes
    node_count = model.cbGet(GRB.Callback.MIPNODE_NODCNT)
    # Heuristic: Limit cuts at deeper nodes to avoid over-cutting
    max_cuts = 5 if node_count < 100 else 2

    digraph = model._graph.copy()
    x = model._x
    y = model._y

    # Label all arcs with weight w_ij = 1 - x_ij
    for i, j in digraph.edges():
        val = min(1.0, model.cbGetNodeRel(x[i, j]))  # cap weight at 1
        digraph[i][j]['weight'] = 1 - val

    # Find violations but be selective about which to add
    cuts_added = 0
    for u, v, data in digraph.edges(data=True):
        # prevent excessively adding cutting plains
        if cuts_added >= max_cuts:
            break

        # only search for cycles connecting nodes that are actually part of the current k-MST solution
        u_included = model.cbGetNodeRel(y[u]) > EPSILON
        v_included = model.cbGetNodeRel(y[v]) > EPSILON
        if not (u_included and v_included):
            continue

        # For each arc (u,v), find the cheapest path from v to u w.r.t weights w_ij
        path_cost, shortest_path = nx.single_source_dijkstra(digraph, source=v, target=u, weight='weight')
        total_cost = path_cost + data['weight']

        # If any shortest path plus arc (u,v) has total weight <1, record a violation
        if total_cost < 1.0 - EPSILON:
            violation_severity = 1.0 - total_cost  # by how much is the cycle constraint violated
            # Only add constraints that are violated significantly
            if violation_severity >= 0.4:
                edges_in_path = [(shortest_path[i], shortest_path[i + 1]) for i in range(len(shortest_path) - 1)]
                cycle = edges_in_path + [(u, v)]

                # add the cycle inequality as a cutting plain
                model.cbCut(gp.quicksum(x[i, j] for i, j in cycle) <= len(cycle) - 1)
                cuts_added += 1


def find_violated_dcc_float(model: gp.Model):
    # Check how deep we are in exploring LP nodes
    node_count = model.cbGet(GRB.Callback.MIPNODE_NODCNT)
    # Heuristic: Limit cuts at deeper nodes to avoid over-cutting
    max_cuts = 5 if node_count < 100 else 2

    digraph = model._graph.copy()
    digraph_with_zero = model._digraph_with_zero.copy()
    x = model._x
    y = model._y
    source_vertex = 0

    cuts_added = 0
    # Label all arcs with weight w_ij = x_ij
    for i, j in digraph_with_zero.edges():
        x_var = model.cbGetNodeRel(x[i, j])
        digraph_with_zero[i][j]['weight'] = min(1.0, max(x_var, 0.0))

    for target_node in digraph.nodes():
        # prevent excessively adding cutting plains
        if cuts_added >= max_cuts:
            break

        y_value = model.cbGetNodeRel(y[target_node])
        # trivial case: if y is 0 anyway (or close to it), we don't care about this vertex
        if y_value <= EPSILON:
            continue

        cut_val, (s, t) = nx.minimum_cut(digraph_with_zero, _s=source_vertex, _t=target_node, capacity='weight')
        if cut_val < y_value - EPSILON and target_node in t:
            violation_severity = 1.0 - cut_val  # by how much is the connectivity constraint violated
            # Only add constraints that are violated significantly
            if violation_severity >= 0.4:
                # add the connectivity inequality as a cutting plain
                model.cbCut(
                    gp.quicksum(x[u, v] for u, v in digraph_with_zero.edges() if u in s and v in t)
                    >= y[target_node]
                )
                cuts_added += 1