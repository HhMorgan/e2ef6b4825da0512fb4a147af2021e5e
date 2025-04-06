import argparse
import os

from pathlib import Path

import gurobipy as gp
import networkx as nx
from gurobipy import GRB

from src.util.utils import generate_names, generate_name_matrix, generate_var_map, generate_var_map_constant, \
    add_variables, add_constraint, set_value, get_variable


def read_instance_file(filename: str | os.PathLike) -> nx.Graph:
    with open(Path(filename), mode="r", encoding="utf-8") as f:
        n_nodes = int(f.readline())
        n_edges = int(f.readline())

        graph = nx.Graph()

        # skip comment line
        f.readline()

        # read node lines
        for _ in range(n_nodes):
            line = f.readline()
            node_id, name, supply_demand = line.split()
            graph.add_node(int(node_id), name=name, supply_demand=int(supply_demand))

        # skip comment line
        f.readline()

        # read edge lines
        for _ in range(n_edges):
            line = f.readline()
            (
                edge_id,
                node_1,
                node_2,
                transport_cost,
                build_cost_1,
                build_cost_2,
                capacity_1,
                capacity_2,
            ) = line.split()
            graph.add_edge(
                int(node_1),
                int(node_2),
                id=int(edge_id),
                transport_cost=int(transport_cost),
                build_cost_1=int(build_cost_1),
                build_cost_2=int(build_cost_2),
                capacity_1=int(capacity_1),
                capacity_2=int(capacity_2),
            )

        return graph


def build_model(model: gp.Model, graph: nx.Graph):
    n = len(graph.nodes)

    complete_graph = {(i, j) for i in range(1, n+1) for j in range(1, n+1) if i != j}  # set comprehension
    reversed_edges = {(j,i) for (i,j) in graph.edges}
    present_edges = reversed_edges.union(set(graph.edges))
    complement = complete_graph.difference(present_edges)


    # given variables
    demand_names = generate_names(n, "b")
    trans_cost_names = generate_name_matrix(1, n + 1, n, n, "c")
    building_cost_1_names = generate_name_matrix(1, n + 1, n, n, "d1")
    building_cost_2_names = generate_name_matrix(1, n + 1, n, n, "d2")
    capacity_1_names = generate_name_matrix(1, n + 1, n, n, "u1")
    capacity_2_names = generate_name_matrix(1, n + 1, n, n, "u2")

    demand_vars = generate_var_map_constant(demand_names, GRB.CONTINUOUS, -GRB.INFINITY, GRB.INFINITY)
    trans_cost_vars = generate_var_map_constant(trans_cost_names, GRB.CONTINUOUS, 0, GRB.INFINITY)
    building_cost_1_vars = generate_var_map_constant(building_cost_1_names, GRB.CONTINUOUS, 0, GRB.INFINITY)
    building_cost_2_vars = generate_var_map_constant(building_cost_2_names, GRB.CONTINUOUS, 0, GRB.INFINITY)
    capacity_1_vars = generate_var_map_constant(capacity_1_names, GRB.CONTINUOUS, 0, GRB.INFINITY)
    capacity_2_vars = generate_var_map_constant(capacity_2_names, GRB.CONTINUOUS, 0, GRB.INFINITY)

    # optimization variables
    flows_names = generate_name_matrix(1, n + 1, n, n, "f")
    is_1_built_names = generate_name_matrix(1, n + 1, n, n, "x1")
    is_2_built_names = generate_name_matrix(1, n + 1, n, n, "x2")

    flow_vars = generate_var_map_constant(flows_names, GRB.CONTINUOUS , 0, GRB.INFINITY)
    built_link_1_vars = generate_var_map_constant(is_1_built_names, GRB.BINARY, 0, 1)
    built_link_2_vars = generate_var_map_constant(is_2_built_names, GRB.BINARY , 0, 1)


    add_variables(model, demand_vars)
    add_variables(model, trans_cost_vars)
    add_variables(model, building_cost_1_vars)
    add_variables(model, building_cost_2_vars)
    add_variables(model, capacity_1_vars)
    add_variables(model, capacity_2_vars)

    add_variables(model, flow_vars)
    add_variables(model, built_link_1_vars)
    add_variables(model, built_link_2_vars)

    model.update() # update the model with all the added variables

    for i in range(1, n+1):
        set_value(model, "b_" + str(i), graph.nodes[i]['supply_demand'])
        b_i = get_variable(model, "b_" + str(i))

        sum_incoming = gp.quicksum(get_variable(model, "f_" + str(i) + "_" + str(j)) for j in range(1, n+1))
        sum_outgoing = gp.quicksum(get_variable(model, "f_" + str(j) + "_" + str(i)) for j in range(1, n+1))
        add_constraint(model, "flow_conservation_" + str(i), sum_incoming - sum_outgoing == b_i)

    for (i, j) in graph.edges:
        props = graph.edges[i, j]
        create_edge_constraints(i, j, props)
        create_edge_constraints(j, i, props)

    # prevent missing edges from being built
    for (i, j) in complement:
        ij_suffix = str(i) + "_" + str(j)
        x1_ij = get_variable(model, "x1_" + ij_suffix)
        x2_ij = get_variable(model, "x2_" + ij_suffix)
        f_ij = get_variable(model, "f_" + ij_suffix)

        add_constraint(model, "unbuildable_edge" + ij_suffix, x1_ij + x2_ij == 0)
        add_constraint(model, "unusable_flow" + ij_suffix, f_ij == 0)

    # objective function
    objective_func = gp.quicksum(
        get_variable(model, "f_" + str(i) + "_" + str(j)) *
        get_variable(model, "c_" + str(i) + "_" + str(j)) +
        get_variable(model, "x1_" + str(i) + "_" + str(j)) *
        get_variable(model, "d1_" + str(i) + "_" + str(j)) +
        get_variable(model, "x2_" + str(i) + "_" + str(j)) *
        get_variable(model, "d2_" + str(i) + "_" + str(j))
            for (i,j) in complete_graph)

    model.setObjective(objective_func, GRB.MINIMIZE)

def create_edge_constraints(i: int, j: int, props: dict):
    ij_suffix = str(i) + "_" + str(j)

    # variables
    set_value(model, "c_" + ij_suffix, props['transport_cost'])
    set_value(model, "d1_" + ij_suffix, props['build_cost_1'])
    set_value(model, "d2_" + ij_suffix, props['build_cost_2'])
    set_value(model, "u1_" + ij_suffix, props['capacity_1'])
    set_value(model, "u2_" + ij_suffix, props['capacity_2'])

    # constraints
    x1_ij = get_variable(model, "x1_" + ij_suffix)
    x2_ij = get_variable(model, "x2_" + ij_suffix)
    u1_ij = get_variable(model, "u1_" + ij_suffix)
    u2_ij = get_variable(model, "u2_" + ij_suffix)
    f_ij = get_variable(model, "f_" + ij_suffix)

    add_constraint(model, "only_build_one" + ij_suffix, x1_ij + x2_ij <= 1)
    add_constraint(model, "constrained_flow" + ij_suffix, f_ij <= x1_ij * u1_ij + x2_ij * u2_ij)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", default="../../instances/ex1.1-instance.dat")
    args = parser.parse_args()

    graph = read_instance_file(args.filename)

    model = gp.Model("ex1.1")
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

    model.close()
