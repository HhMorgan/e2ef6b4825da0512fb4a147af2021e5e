import gurobipy as gp
from gurobipy import Var

def add_variable(model: gp.Model, name: str, vtype, lower, upper) -> Var:
        return model.addVar(vtype=vtype, name=name, lb=lower, ub=upper)

def add_variables(model, var_map=None) -> list:
    if var_map is None:
        var_map = {}

    var_list = []
    for name, props in var_map.items():
        var_list.append(add_variable(model, name, props["vtype"], props["lower"], props["upper"]))

    return var_list

def generate_names(number: int, prefix) -> list:
    names_list = []
    for i in range(number):
        names_list.append(prefix + "_" + str(i))

    return names_list

def generate_name_matrix(row_num, col_num, prefix) -> list:
    names_list = []
    for i in range(row_num):
        for j in range(col_num):
            names_list.append(prefix + "_" + str(i) + "_" + str(j))

    return names_list

def add_constraint(model: gp.Model, name: str, inequality):
    return model.addConstraint(inequality, name=name)
