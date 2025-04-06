import gurobipy as gp
from gurobipy import Var, GRB

TYPE = "vtype"
LOWER = "lower"
UPPER = "upper"


def add_variable(model: gp.Model, name: str, vtype, lower, upper) -> Var:
        return model.addVar(vtype=vtype, name=name, lb=lower, ub=upper)

def add_variable_unconstrained(model: gp.Model, name: str, vtype) -> Var:
    return model.addVar(vtype=vtype, name=name)

def add_variables(model, var_map=None) -> list:
    if var_map is None:
        var_map = {}

    var_list = []
    for name, props in var_map.items():
        if props[TYPE] is GRB.BINARY:
            var_list.append(add_variable_unconstrained(model, name, GRB.BINARY))
        else:
            var_list.append(add_variable(model, name, props[TYPE], props[LOWER], props[UPPER]))

    return var_list

def generate_names(number: int, prefix) -> list:
    names_list = []
    for i in range(number):
        names_list.append(prefix + "_" + str(i + 1))

    return names_list

def generate_names_in_range(from_idx: int, to_idx: int, prefix) -> list:
    names_list = []
    for i in range(from_idx, to_idx):
        names_list.append(prefix + "_" + str(i))

    return names_list

def generate_name_matrix(from_idx: int, to_idx: int,row_num, col_num, prefix) -> list:
    names_list = []
    for i in range(from_idx, to_idx):
        for j in range(1, col_num + 1):
            names_list.append(prefix + "_" + str(i) + "_" + str(j))

    return names_list

def generate_var_map(names: list, types: list, lower_bounds: list, upper_bounds: list) -> dict:
    if len(names) != len(types) or len(names) != len(lower_bounds) or len(names) != len(upper_bounds):
        raise Exception("Incompatible number of names, types and bounds!")

    var_map = {}
    for i in range(len(names)):
        props = {
            TYPE: types[i],
            LOWER: lower_bounds[i],
            UPPER: upper_bounds[i]
        }
        var_map[names[i]] = props

    return var_map


def generate_var_map_constant(names: list, type, lower_bound, upper_bound) -> dict:
    var_map = {}
    for i in range(len(names)):
        props = {
            TYPE: type,
            LOWER: lower_bound,
            UPPER: upper_bound
        }
        var_map[names[i]] = props

    return var_map

def add_constraint(model: gp.Model, name: str, inequality):
    return model.addConstr(inequality, name=name)


def get_variable(model: gp.Model, name: str) -> Var:
    return model.getVarByName(name)

def set_value(model: gp.Model, var_name: str, value):
    variable = get_variable(model, var_name)
    model.addConstr(variable == value, name=var_name + "_value")