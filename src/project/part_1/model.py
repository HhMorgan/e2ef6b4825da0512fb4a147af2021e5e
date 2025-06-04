import gurobipy as gp
import networkx as nx
from gurobipy import GRB

TOLERANCE: float = 1e-05


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
            # find_violated_dcc_float(model)
            kawai = 1


def find_violated_cec_int(model: gp.Model):
    # Create a graph including all edges who's x-value is (roughly) 1
    x = model._x
    G = nx.Graph()
    for (i, j), x_var in x.items():
        val = model.cbGetSolution(x_var)
        if val > 1 - TOLERANCE:
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
    # find_violated_dcc_float(model)
    digraph_with_zero = model._digraph_with_zero.copy()
    digraph = model._graph.copy()
    x = model._x
    y = model._y
    source_vertex = 0
    EPSILON = 1e-6  # Small positive value
    # Label all arcs with weight w_ij = 1 - x_ij
    for i, j in digraph_with_zero.edges():
        x_var = model.cbGetSolution(x[i, j])
        # val = max(x_var, EPSILON)  # Ensure weight is always positive
        # print(val)
        digraph_with_zero[i][j]['weight'] = x_var
    for node in digraph.nodes():
        y_var = model.cbGetSolution(y[node])
        # if node == source_vertex:
        #     continue
        cut_val, partition = nx.minimum_cut(digraph_with_zero, source_vertex, node, capacity='weight')
        # print(cut_val, partition)
        print(y_var)
        if cut_val < 2 - TOLERANCE:
            model.cbLazy(
                gp.quicksum(x[u, v] for u, v in digraph.edges() if u in partition[0] and v in partition[1]) >= y_var)
    # return dcc

    pass


#======================--------=============================
def find_violated_cec_float(model: gp.Model):
    # Check how deep we are in exploring LP nodes
    node_count = model.cbGet(GRB.Callback.MIPNODE_NODCNT)
    # Heuristic: Limit cuts at deeper nodes to avoid over-cutting
    max_cuts = 4 if node_count < 100 else 2

    digraph = model._graph.copy()
    x = model._x

    # Store violations and their effectiveness
    violations = []

    # Label all arcs with weight w_ij = 1 - x_ij
    for i, j in digraph.edges():
        val = min(1.0, model.cbGetNodeRel(x[i, j])) # cap weight at 1
        digraph[i][j]['weight'] = 1 - val

    # Find violations but be selective about which to add
    for u, v, data in digraph.edges(data=True):
        # prevent excessively adding cutting plains
        if len(violations) >= max_cuts:
            break

        # For each arc (u,v), find the cheapest path from v to u w.r.t weights w_ij
        path_cost, shortest_path = nx.single_source_dijkstra(digraph, source=v, target=u, weight='weight')
        total_cost = path_cost + data['weight']

        # If any shortest path plus arc (u,v) has total weight <1, record a violation
        if total_cost < 1.0 - TOLERANCE:
            violation_severity = 1.0 - total_cost # by how much is the cycle constraint violated
            edges_in_path = [(shortest_path[i], shortest_path[i + 1]) for i in range(len(shortest_path) - 1)]

            # Store violation with its effectiveness score
            violations.append({
                'cycle': edges_in_path + [(u,v)],
                'severity': violation_severity,
            })

    # Sort by violation severity (most violated inequality first)
    violations.sort(key=lambda x: x['severity'], reverse=True)

    # Add only the most violated constraints
    cuts_added = 0
    for viol in violations[:max_cuts]:
        cycle = viol['cycle']

        # add the cycle inequality as a cutting plain
        model.cbCut(gp.quicksum(x[i, j] for i, j in cycle) <= len(cycle) - 1)
        cuts_added += 1
        # print(f"Added inequality number {cuts_added}!")


def find_violated_dcc_float(model: gp.Model):
    # TODO Something about finding a minimum cut (see slides) - use networkx mincut function for this
    digraph = model._graph.copy()
    x = model._x
    source_vertex = list(digraph.nodes())[0]
    # Label all arcs with weight w_ij = 1 - x_ij
    for i, j in digraph.edges():
        val = min(1.0, model.cbGetNodeRel(x[i, j]))  # cap weight at 1
        digraph[i][j]['weight'] = val
    for node in digraph.nodes():
        cut_val, partition = nx.minimum_cut(digraph, source_vertex, node, capacity='weight')
        if cut_val < 2 - TOLERANCE:
            model.cbLazy( gp.quicksum(x[u, v] for u, v in digraph.edges() if u in partition[0] and v in partition[1]) >= 1)
    # return dcc
    pass


def create_model(model: gp.Model, graph: nx.Graph, k: int, *, digraph: nx.Graph = None):
    node_indices = [n for n in graph]  # grab the ID of every node in the graph
    # print(graph.nodes)
    # print(list(graph.nodes())[0])
    reversed_arcs = {(j, i) for (i, j) in graph.edges}
    arcs = reversed_arcs.union(set(graph.edges))  # edges in graph and their inverted counterpart
    arcs_with_zero = arcs.union((0, j) for j in node_indices)

    # create a directed NX graph from all the arcs (useful for querying incident arcs later)
    if graph.is_directed():
        digraph = graph
    elif not digraph:
        digraph = nx.DiGraph()
        digraph.add_edges_from(arcs)

    digraph_with_zero = digraph.copy()
    digraph_with_zero.add_node(0)
    digraph_with_zero.add_edges_from((0, j) for j in node_indices)

    model._graph = digraph  # add reference to the initial directed graph for use in CEC and DCC formulations
    model._digraph_with_zero = digraph_with_zero

    # common variables
    y = model.addVars(
        node_indices,
        name="y",
        vtype=GRB.BINARY,
    )
    x = model.addVars(
        arcs_with_zero,
        name="x",
        vtype=GRB.BINARY
    )

    # add reference to relevant variables for later use in callbacks (CEC,DCC)
    model._x = x
    model._y = y

    # create model-specific variables and constraints
    if model._formulation == "seq":
        v = model.addVars(
            range(0, len(node_indices) + 1),
            name="v",
            vtype=GRB.CONTINUOUS,
        )

        model.addConstrs((v[i] + x[i, j] <= v[j] + k * (1 - x[i, j])
                          for (i, j) in arcs_with_zero),
                         "impose_order")
        model.addConstrs((gp.quicksum(x[i, j] for i in digraph_with_zero.predecessors(j)) == y[j]
                          for j in node_indices),
                         "one_incoming_edge")
        model.addConstrs((y[i] + y[j] >= 2 * x[i, j] for (i, j) in arcs),
                         "edge_implies_vertices")

        model.addConstr(gp.quicksum(x[0, j] for j in node_indices) == 1, name="one_edge_from_root")
        # constraints for root node 0
        model.addConstr(v[0] == 0, "zero_is_root")

        model.addConstr(gp.quicksum(x[i, j] for (i, j) in arcs) == k - 1,
                        "k_vertices")
        # define restricted interval for order variable
        model.addConstrs((v[i] >= y[i] for i in node_indices),
                         "positive_ordering")
        model.addConstrs((v[i] <= k * y[i] for i in node_indices),
                         "integer-step_ordering")


    elif model._formulation == "scf":
        f = model.addVars(
            arcs_with_zero,
            name="f",
            lb=0,
            vtype=GRB.CONTINUOUS,
        )

        model.addConstr(gp.quicksum(f[0, j] for j in node_indices) == k, name="source_flow")

        model.addConstr(gp.quicksum(x[0, j] for j in node_indices) == 1, name="one_edge_from_root")

        model.addConstrs((gp.quicksum(f[i, j] for i in digraph_with_zero.predecessors(j)) -
                          gp.quicksum(f[j, i] for i in digraph_with_zero.successors(j)) == y[j]
                          for j in node_indices),
                         name="consume_one_unit")

        model.addConstrs(
            (y[j] >= 1 / len(node_indices) * gp.quicksum(x[i, j] for i in digraph_with_zero.predecessors(j))
             for j in node_indices),
            name="flow_inclusion")
        model.addConstrs((y[j] <= gp.quicksum(x[i, j] for i in digraph_with_zero.predecessors(j))
                          for j in node_indices),
                         name="flow_exclusion")

        # hint: positive flow constraint is managed through lower bound for f variable
        model.addConstrs((f[i, j] <= k * x[i, j]
                          for (i, j) in arcs_with_zero),
                         name="capped_flow")


    elif model._formulation == "mcf":
        arcs_times_vertices = set((i, j, v) for (i, j) in arcs for v in node_indices)
        arcs_times_vertices_with_zero = arcs_times_vertices.union((0, j, v) for j in node_indices for v in node_indices)

        f = model.addVars(
            arcs_times_vertices_with_zero,
            name="f",
            lb=0,
            vtype=GRB.CONTINUOUS,
        )

        model.addConstrs((gp.quicksum(f[0, j, v] for j in node_indices) <= y[v]
                          for v in node_indices), name="source_flow")

        model.addConstr(gp.quicksum(x[0, j] for j in node_indices) == 1, name="one_edge_from_root")

        model.addConstrs((gp.quicksum(f[i, v, v] for i in digraph_with_zero.predecessors(v)) -
                          gp.quicksum(f[v, j, v] for j in digraph_with_zero.successors(v)) == y[v]
                          for v in node_indices), "consume_own_flow")

        model.addConstrs((gp.quicksum(f[i, j, v] for i in digraph_with_zero.predecessors(j)) -
                          gp.quicksum(f[j, i, v] for i in digraph_with_zero.successors(j))
                          == 0
                          for j in node_indices for v in node_indices if v != j),
                         name="non-consumption_foreign_flow")

        model.addConstrs((y[i] + y[j] >= 2 * x[i, j] for (i, j) in arcs),
                         "edge_implies_vertices")

        model.addConstrs((gp.quicksum(x[i, j] for i in digraph_with_zero.predecessors(j)) == y[j]
                          for j in node_indices),
                         "one_incoming_edge")

        # hint: positive flow constraint is managed through lower bound for f variable
        model.addConstrs((f[i, j, v] <= x[i, j]
                          for (i, j, v) in arcs_times_vertices_with_zero),
                         name="unit_flow")

        model.addConstr(gp.quicksum(x[i, j] for (i, j) in arcs) == k - 1,
                        name="take_k_edges")


    elif model._formulation == "cec":
        model.addConstrs((y[i] + y[j] >= 2 * x[i, j] for (i, j) in arcs),
                         "edge_implies_vertices")

        model.addConstrs((x[i, j] + x[j, i] <= 1 for (i, j) in arcs if i < j),
                         "edge_one_direction")

        model.addConstr(gp.quicksum(x[i, j] for (i, j) in arcs) == k - 1,
                        "k_1_edges")

        model.addConstr(gp.quicksum(x[0, j] for j in node_indices) == 1, name="one_edge_from_root")

        # note: This is already covered by "edge_implies_vertices" and "one_incoming_edge"
        # model.addConstr(gp.quicksum(y[i] for i in node_indices) == k,
        #                 "k_vertices")
        model.addConstrs((gp.quicksum(x[i, j] for i in digraph_with_zero.predecessors(j)) == y[j]
                          for j in node_indices),
                         "one_incoming_edge")

    elif model._formulation == "dcc":
        # TODO Implement DCC formulation
        model.addConstrs((y[i] + y[j] >= 2 * x[i, j] for (i, j) in arcs),
                         "edge_implies_vertices")

        model.addConstrs((x[i, j] + x[j, i] <= 1 for (i, j) in arcs if i < j),
                         "edge_one_direction")

        model.addConstr(gp.quicksum(x[i, j] for (i, j) in arcs) == k - 1,
                        "k_1_edges")

        model.addConstr(gp.quicksum(x[0, j] for j in node_indices) == 1, name="one_edge_from_root")

        model.addConstrs((gp.quicksum(x[i, j] for i in digraph_with_zero.predecessors(j)) == y[j]
                          for j in node_indices),
                         "one_incoming_edge")

    # common objective function
    model.setObjective(gp.quicksum(x[i, j] * graph.edges[i, j]['cost'] for (i, j) in arcs), GRB.MINIMIZE)


def get_selected_edge_ids(model: gp.Model, graph: nx.Graph) -> list[int]:
    # note that you may need to account for tolerances
    # see, e.g., https://docs.gurobi.com/projects/optimizer/en/current/concepts/modeling/tolerances.html

    # https://docs.gurobi.com/projects/optimizer/en/current/concepts/attributes/examples.html

    reversed_arcs = {(j, i) for (i, j) in graph.edges}
    arcs = reversed_arcs.union(set(graph.edges))  # edges in graph and their inverted counterpart

    selected_edges: list[int] = []
    if model.SolCount > 0:
        # for v in sorted(model.getVars(), key=lambda x: x.VarName):
        #     print(f"{v.VarName:<8} = {v.X}")

        for (i, j) in arcs:
            x_ij = model.getVarByName(f'x[{i},{j}]')

            if x_ij.X == 1:
                edge_id: int = int(graph.edges[i, j]['id'])
                selected_edges.append(edge_id)
    # else:
    #     for v in sorted(model.getVars(), key=lambda x: x.VarName):
    #         print(v.VarName)

    return selected_edges
