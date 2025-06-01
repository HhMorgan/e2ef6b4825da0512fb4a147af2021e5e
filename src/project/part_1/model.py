import gurobipy as gp
import networkx as nx
from gurobipy import GRB


def lazy_constraint_callback(model: gp.Model, where):
    # note: you'll need to account for tolerances!
    # see, e.g., https://docs.gurobi.com/projects/optimizer/en/current/concepts/modeling/tolerances.html

    # check integer solutions for feasibility
    if where == GRB.Callback.MIPSOL:
        # get solution values for variables x
        # see https://docs.gurobi.com/projects/optimizer/en/current/reference/python/model.html#Model.cbGetSolution

        if model._formulation == "cec":
            inequality = find_violated_cec_int(model)
        elif model._formulation == "dcc":
            inequality = find_violated_dcc_int(model)

    # check fractional solutions to find violated CECs/DCCs to strengthen the bound
    elif where == GRB.Callback.MIPNODE and model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
        # get solution values for variables x
        # see https://docs.gurobi.com/projects/optimizer/en/current/reference/python/model.html#Model.cbGetNodeRel

        # you may also use different algorithms for integer and fractional separation if you want
        if model._formulation == "cec":
            inequality = find_violated_cec_float(model)
        elif model._formulation == "dcc":
            inequality = find_violated_dcc_float(model)

    # TODO In general: Add a cutting plane (strengthening inequality)
    # x = model.cbGetNodeRel(model._x)
    # rel = model.cbGetNodeRel(x)
    # if rel[0] + rel[1] > 1.1:
    #     model.cbCut(inequality)
    #     model.cbCut(x[0] + x[1], GRB.LESS_EQUAL, 1)


def find_violated_cec_int(model: gp.Model):
    # TODO Create subgraph with chosen arcs {(i,j) | x_ij = 1}

    # TODO Find a cycle in this subgraph and return a CEC for it
    pass


def find_violated_dcc_int(model: gp.Model):
    # add your DCC separation code here
    pass


def find_violated_cec_float(model: gp.Model):
    # TODO Label all arcs in the digraph with weight w_ij = 1 - x_ij

    # TODO For each arc (u,v), find cheapest path from v to u w.r.t weights w_ij (use networkx method like Dijkstra)
    # for (i,j in ...:
    # TODO If any shortest path plus arc (u,v) has total weight 0, create a cycle-elimination constraint from this
    # return cec
    pass


def find_violated_dcc_float(model: gp.Model):
    # TODO Something about finding a minimum cut (see slides) - use networkx mincut function for this

    # return dcc
    pass


def create_model(model: gp.Model, graph: nx.Graph, k: int, *, digraph: nx.Graph = None):
    node_indices = [n for n in graph]  # grab the ID of every node in the graph

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

    # common variables
    x = model.addVars(
        arcs_with_zero,
        name="x",
        vtype=GRB.BINARY
    )
    y = model.addVars(
        node_indices,
        name="y",
        vtype=GRB.BINARY,
    )

    # add reference to relevant variables for later use in callbacks (CEC,DCC)
    model._x = x
    model._y = y

    # common constraints
    model.addConstr(gp.quicksum(x[0, j] for j in node_indices) == 1, name="one_edge_from_root")

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
        # TODO Implement CEC formulation

        model.addConstr(gp.quicksum(x[i, j] for (i, j) in arcs) == k - 1, "k_vertices")

    elif model._formulation == "dcc":
        # TODO Implement DCC formulation

        model.addConstr(gp.quicksum(x[i, j] for (i, j) in arcs) == k - 1, "k_vertices")

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
