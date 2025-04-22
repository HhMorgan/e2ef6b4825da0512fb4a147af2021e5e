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
    n = len(node_indices)

    reversed_edges = {(j, i) for (i, j) in graph.edges}
    present_edges = reversed_edges.union(set(graph.edges))  # edges in graph and their inverted counterpart
    present_edges_with_root = present_edges.union((0, j) for j in node_indices)

    # create common variables
    # see, e.g., https://docs.gurobi.com/projects/optimizer/en/current/reference/python/model.html#Model.addVars

    x = model.addVars(
        present_edges_with_root,
        name="x",
        vtype=GRB.BINARY
    )

    # add reference to relevant variables for later use in callbacks (CEC,DCC)
    model._x = x


    # create common constraints
    # see, e.g., https://docs.gurobi.com/projects/optimizer/en/current/reference/python/model.html#Model.addConstr


    # create model-specific variables and constraints
    if model._formulation == "seq":
        v = model.addVars(
            node_indices,
            name="v",
            vtype=GRB.INTEGER,
        )

        # TODO Think about how to change the x_e variable constraint to a directed formulation
        model.addConstrs((v[i] >= 1 for i in node_indices), "positive ordering")
        model.addConstrs((v[i] <= node_indices[-1] for i in node_indices), "integer-step ordering")
        model.addConstr(v[0] == 0, "zero vertex as root node")
        model.addConstr(gp.quicksum(x[i,j] for (i,j) in present_edges) == k, "MST with k vertices")

    elif model._formulation == "scf":
        v = model.addVars(
            node_indices,
            name="v",
            vtype=GRB.BINARY,
        )

        f = model.addVars(
            present_edges.union((0,j) for j in node_indices),
            name="f",
            vtype=GRB.INTEGER,
        )

        model.addConstr(gp.quicksum(f[0, j] for j in node_indices) == k, name="source flow")
        model.addConstr(gp.quicksum(x[0, j] for j in node_indices) == 1, name="only one edge from root")

        model.addConstrs((gp.quicksum(f[i, j] for (i, j) in present_edges_with_root if j == j_n) -
                          gp.quicksum(f[j, i] for (j, i) in present_edges_with_root if j == j_n) == v[j_n]
                          for j_n in node_indices),
                          name="consume one unit")

        model.addConstrs((v[n] >= 1/n * gp.quicksum(x[i,j] for (i,j) in present_edges_with_root if j == n)
                          for n in node_indices), name="be present if you receive flow")
        model.addConstrs((v[n] <= gp.quicksum(x[i,j] + x[j,i] for (i,j) in present_edges if j == n)
                          for n in node_indices), name="be absent if you receive no flow")

        model.addConstrs((0 <= f[i,j] for (i, j) in present_edges_with_root), name="positive flow")
        model.addConstrs((f[i,j] <= k * x[i,j] for (i, j) in present_edges_with_root), name="capped flow")

    elif model._formulation == "mcf":
        present_edges_times_vertices = set((i,j,v) for (i,j) in present_edges for v in node_indices)

        f = model.addVars(
            present_edges_times_vertices.union((0,j,v) for j in node_indices for v in node_indices),
            name="f",
            vtype=GRB.BINARY,
        )

        model.addConstrs((gp.quicksum(f[0, j, v] for j in node_indices) == 1
                        for v in node_indices), name="source flow")

        model.addConstrs((gp.quicksum(f[i,v,v] for (i,v) in present_edges if v == v_n) ==
                         1/n * gp.quicksum(x[i,v] for (i,v) in present_edges if v == v_n)
                         for v_n in node_indices), "consume own flow")

        model.addConstrs((gp.quicksum(f[i,j,v] for (i,j,v) in present_edges_times_vertices if j == j_n and v == v_n) -
                          gp.quicksum(f[j,i,v] for (i,j,v) in present_edges_times_vertices if j == j_n and v == v_n)
                          == 0
                          for j_n in node_indices for v_n in node_indices if v_n != j_n),
                         name="non-consumption of foreign flow")

        model.addConstrs((0 <= f[i,j,v] for (i,j,v) in present_edges_times_vertices),
                         name="positive flow")
        model.addConstrs((f[i,j,v] <= x[i,j] for (i,j,v) in present_edges_times_vertices),
                         name="unit flow")
        model.addConstr(gp.quicksum(x[i,j] for (i,j) in present_edges) == k-1,
                        name="take k-1 edges")

    elif model._formulation == "cec":
        pass
    elif model._formulation == "dcc":
        pass

    # common objective function
    model.setObjective(gp.quicksum(x[i,j] * graph.edges[i,j]['cost'] for (i,j) in present_edges), GRB.MINIMIZE)


def get_selected_edge_ids(model: gp.Model, graph: nx.Graph) -> list[int]:
    # note that you may need to account for tolerances
    # see, e.g., https://docs.gurobi.com/projects/optimizer/en/current/concepts/modeling/tolerances.html

    # https://docs.gurobi.com/projects/optimizer/en/current/concepts/attributes/examples.html



    reversed_edges = {(j, i) for (i, j) in graph.edges}
    present_edges = reversed_edges.union(set(graph.edges))  # edges in graph and their inverted counterpart

    selected_edges: list[int] = []
    if model.SolCount > 0:
        for v in sorted(model.getVars(), key=lambda x: x.VarName):
            print(v.VarName, v.X)

        for (i,j) in present_edges:
            x_ij = model.getVarByName(f'x[{i},{j}]')

            if x_ij.X == 1:
                edge_id: int = int(graph.edges[i,j]['id'])
                selected_edges.append(edge_id)
    else:
        for v in sorted(model.getVars(), key=lambda x: x.VarName):
            print(v.VarName)

    return selected_edges