import argparse
import os
from pathlib import Path

import gurobipy as gp
import networkx as nx
from gurobipy import GRB

from src.util.utils import generate_names, generate_name_matrix, generate_var_map, generate_var_map_constant, \
    add_variables, add_constraint, set_value


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
    m = len(graph.edges)

    # given variables
    demand_names = generate_names(n, "b")
    trans_cost_names = generate_name_matrix(n, n, "c")
    building_cost_1_names = generate_name_matrix(n, n, "d1")
    building_cost_2_names = generate_name_matrix(n, n, "d2")
    capacity_1_names = generate_name_matrix(n, n, "u1")
    capacity_2_names = generate_name_matrix(n, n, "u2")

    demand_vars = generate_var_map_constant(demand_names, GRB.CONTINUOUS , GRB.INFINITY, -GRB.INFINITY)
    trans_cost_vars = generate_var_map_constant(trans_cost_names, GRB.CONTINUOUS , GRB.INFINITY, 0)
    building_cost_1_vars = generate_var_map_constant(building_cost_1_names, GRB.CONTINUOUS , GRB.INFINITY, 0)
    building_cost_2_vars = generate_var_map_constant(building_cost_2_names, GRB.CONTINUOUS , GRB.INFINITY, 0)
    capacity_1_vars = generate_var_map_constant(capacity_1_names, GRB.CONTINUOUS , GRB.INFINITY, 0)
    capacity_2_vars = generate_var_map_constant(capacity_2_names, GRB.CONTINUOUS , GRB.INFINITY, 0)

    # optimization variables
    flows_names = generate_name_matrix(n, n, "f")
    is_1_built_names = generate_name_matrix(n, n, "x1")
    is_2_built_names = generate_name_matrix(n, n, "x2")

    flow_vars = generate_var_map_constant(flows_names, GRB.CONTINUOUS , GRB.INFINITY, -GRB.INFINITY)
    built_link_1_vars = generate_var_map_constant(is_1_built_names, GRB.BINARY, 0, 1)
    built_link_2_vars = generate_var_map_constant(is_2_built_names, GRB.BINARY , 0, 1)


    add_variables(model, demand_vars)
    add_variables(model, trans_cost_vars)
    add_variables(model, building_cost_1_vars)
    add_variables(model, building_cost_2_vars)
    add_variables(model, capacity_1_vars)
    add_variables(model, capacity_2_vars)
    add_variables(model, built_link_1_vars)
    add_variables(model, built_link_2_vars)

    add_variables(model, flow_vars)
    add_variables(model, built_link_1_vars)
    add_variables(model, built_link_2_vars)

    model.update() # update the model with all the added variables

    for i in graph.nodes:
        set_value(model, "b_" + str(i), graph.nodes[i]['supply_demand'])

    for (i, j) in graph.edges:
        props = graph.edges[i, j]
        ij_suffix = str(i) + "_" + str(j)
        set_value(model, "x_" + ij_suffix, props['transport_cost'])
        set_value(model, "x_" + ij_suffix, props['build_cost_1'])
        set_value(model, "x_" + ij_suffix, props['build_cost_2'])
        set_value(model, "x_" + ij_suffix, props['capacity_1'])
        set_value(model, "x_" + ij_suffix, props['capacity_2'])


    # constraints


    model.setObjective(GRB.MINIMIZE)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", default="../../instances/ex1.1-instance.dat")
    args = parser.parse_args()

    graph = read_instance_file(args.filename)

    model = gp.Model("ex1.1")
    build_model(model, graph)

    # model.update()
    # model.optimize()

    if model.SolCount > 0:
        print(f"obj. value = {model.ObjVal}")
        for v in model.getVars():
            print(f"{v.VarName} = {v.X}")

    model.close()
