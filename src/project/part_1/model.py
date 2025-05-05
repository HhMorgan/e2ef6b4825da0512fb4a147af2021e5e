import gurobipy as gp
from gurobipy import GRB
import networkx as nx


def lazy_constraint_callback(model: gp.Model, where):
    # note: you'll need to account for tolerances!
    # see, e.g., https://docs.gurobi.com/projects/optimizer/en/current/concepts/modeling/tolerances.html

    # check integer solutions for feasibility
    if where == GRB.Callback.MIPSOL:
        # get solution values for variables x
        # see https://docs.gurobi.com/projects/optimizer/en/current/reference/python/model.html#Model.cbGetSolution

        # x_values = model.cbGetSolution(model._x)

        if model._formulation == "cec":
            add_violated_cec(model)
        elif model._formulation == "dcc":
            add_violated_dcc(model)

    # check fractional solutions to find violated CECs/DCCs to strengthen the bound
    elif where == GRB.Callback.MIPNODE and model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
        # get solution values for variables x
        # see https://docs.gurobi.com/projects/optimizer/en/current/reference/python/model.html#Model.cbGetNodeRel
        
        # x_values = model.cbGetNodeRel(model._x)

        # you may also use different algorithms for integer and fractional separation if you want
        if model._formulation == "cec":
            add_violated_cec(model)
        elif model._formulation == "dcc":
            add_violated_dcc(model)


def add_violated_cec(model: gp.Model):
    # add your CEC separation code here
    pass


def add_violated_dcc(model: gp.Model):
    # add your DCC separation code here
    pass


def create_model(model: gp.Model, graph: nx.Graph, k: int):
    graph.undirected = True  # make sure the graph is undirected

    node_indices = [n for n in graph] # grab the ID of every node in the graph

    reversed_arcs = {(j, i) for (i, j) in graph.edges}
    arcs = reversed_arcs.union(set(graph.edges))  # edges in graph and their inverted counterpart
    arcs_with_zero = arcs.union((0, j) for j in node_indices)

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
    # model._x = x

    # create model-specific variables and constraints
    if model._formulation == "seq":
        v = model.addVars(
            range(0, len(node_indices) + 1),
            name="v",
            vtype=GRB.CONTINUOUS,
        )

        model.addConstrs((v[i] + x[i,j] <= v[j] + k * (1 - x[i,j])
                          for (i,j) in arcs_with_zero),
                          "impose_order")
        model.addConstrs((gp.quicksum(x[i,j] for (i,j) in arcs_with_zero if j == j_g) == y[j_g]
                          for j_g in node_indices),
                         "one_incoming_edge")
        model.addConstrs((y[i] + y[j] >= 2 * x[i, j] for (i, j) in arcs),
                         "edge_implies_vertices")

        # constraints for root node 0
        model.addConstr(v[0] == 0, "zero_is_root")
        model.addConstr(gp.quicksum(x[0,j] for j in node_indices) == 1,
                        "one_entrypoint")

        model.addConstr(gp.quicksum(x[i,j] for (i,j) in arcs) == k - 1,
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
            vtype=GRB.CONTINUOUS,
        )

        model.addConstr(gp.quicksum(f[0, j] for j in node_indices) == k, name="source_flow")
        model.addConstr(gp.quicksum(x[0, j] for j in node_indices) == 1, name="one_edge_from_root")

        model.addConstrs((gp.quicksum(f[i, j] for (i, j) in arcs_with_zero if j == j_n) -
                          gp.quicksum(f[j, i] for (j, i) in arcs_with_zero if j == j_n) == y[j_n]
                          for j_n in node_indices),
                          name="consume_one_unit")

        model.addConstrs((y[n] >= 1/len(node_indices) * gp.quicksum(x[i,j] for (i,j) in arcs_with_zero if j == n)
                          for n in node_indices),
                          name="flow_inclusion")
        model.addConstrs((y[n] <= gp.quicksum(x[i,j] for (i,j) in arcs_with_zero if j == n)
                          for n in node_indices),
                          name="flow_exclusion")

        model.addConstrs((0 <= f[i,j]
                          for (i, j) in arcs_with_zero),
                          name="positive_flow")
        model.addConstrs((f[i,j] <= k * x[i,j]
                          for (i, j) in arcs_with_zero),
                         name="capped_flow")


    elif model._formulation == "mcf":
        arcs_times_vertices = set((i,j,v) for (i,j) in arcs for v in node_indices)
        arcs_times_vertices_with_zero = arcs_times_vertices.union((0,j,v) for j in node_indices for v in node_indices)

        f = model.addVars(
            arcs_times_vertices_with_zero,
            name="f",
            vtype=GRB.CONTINUOUS,
        )

        model.addConstrs((gp.quicksum(f[0, j, v] for j in node_indices) <= y[v]
                        for v in node_indices), name="source_flow")
        model.addConstr(gp.quicksum(x[0, j] for j in node_indices) == 1, name="one_entry_point")

        model.addConstrs((gp.quicksum(f[i,v,v] for (i,v) in arcs_with_zero if v == v_n) -
                          gp.quicksum(f[v,i,v] for (v,i) in arcs_with_zero if v == v_n) == y[v_n]
                         for v_n in node_indices), "consume_own_flow")

        model.addConstrs((gp.quicksum(f[i,j,v] for (i,j) in arcs_with_zero if j == j_n) -
                          gp.quicksum(f[j,i,v] for (j,i) in arcs_with_zero if j == j_n)
                          == 0
                          for j_n in node_indices for v in node_indices if v != j_n),
                         name="non-consumption_foreign_flow")

        model.addConstrs((y[i] + y[j] >= 2 * x[i, j] for (i, j) in arcs),
                         "edge_implies_vertices")

        model.addConstrs((gp.quicksum(x[i,j] for (i,j) in arcs_with_zero if j == j_g) == y[j_g]
                          for j_g in node_indices),
                         "one_incoming_edge")

        model.addConstrs((0 <= f[i,j,v]
                          for (i,j,v) in arcs_times_vertices_with_zero),
                          name="positive_flow")
        model.addConstrs((f[i,j,v] <= x[i,j]
                          for (i,j,v) in arcs_times_vertices_with_zero),
                          name="unit_flow")

        model.addConstr(gp.quicksum(x[i,j] for (i,j) in arcs) == k-1,
                        name="take_k_edges")


    elif model._formulation == "cec":
        pass
    elif model._formulation == "dcc":
        pass

    # common objective function
    model.setObjective(gp.quicksum(x[i,j] * graph.edges[i,j]['cost'] for (i,j) in arcs), GRB.MINIMIZE)


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

        for (i,j) in arcs:
            x_ij = model.getVarByName(f'x[{i},{j}]')

            if x_ij.X == 1:
                edge_id: int = int(graph.edges[i,j]['id'])
                selected_edges.append(edge_id)
    # else:
    #     for v in sorted(model.getVars(), key=lambda x: x.VarName):
    #         print(v.VarName)

    return selected_edges