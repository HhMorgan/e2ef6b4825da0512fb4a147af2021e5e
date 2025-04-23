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

def generate_name_matrix(from_idx: int, to_idx: int, row_num: int, col_num: int, prefix: str) -> list:
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

import re
def latex_escape(s):
    """
    Escape characters that are special in LaTeX.
    """
    return re.sub(r'([&_#%])', r'\\\1', s)

def generate_latex_table(model, output_file="variable_table.txt"):
    lines = [
        r"\begin{longtable}{|c|c|}",
        r"\hline",
        r"\textbf{Variable} & \textbf{Value} \\",
        r"\hline",
        r"\endfirsthead",
        r"\hline \textbf{Variable} & \textbf{Value} \\ \hline",
        r"\endhead"
    ]

    for var in model.getVars():
        name = latex_escape(var.VarName)
        val = var.X
        lines.append(f"{name} & {val:.4f} \\\\")
        lines.append(r"\hline")

    lines.append(r"\end{longtable}")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def generate_three_tables_per_page(model, vars_per_table=25, output_file="variable_table.txt"):
    vars_list = sorted(model.getVars(), key=lambda v: v.VarName)
    total_vars = len(vars_list)

    tables = []
    for i in range(0, total_vars, vars_per_table):
        chunk = vars_list[i:i + vars_per_table]

        table = [
            r"\begin{tabular}{|c|c|}",
            r"\hline",
            r"\textbf{Variable} & \textbf{Value} \\",
            r"\hline"
        ]

        for var in chunk:
            name = latex_escape(var.VarName)
            val = var.X
            table.append(f"{name} & {val:.4f} \\\\")
            table.append(r"\hline")

        table.append(r"\end{tabular}")
        tables.append("\n".join(table))

    # Group every 3 tables with spacing and page breaks
    final_output = []
    for i in range(0, len(tables), 3):
        group = tables[i:i+3]
        final_output.append(r"\noindent")
        for t in group:
            final_output.append(r"{\small")
            final_output.append(t)
            final_output.append(r"}")
            final_output.append(r"\vspace{1em}")
        final_output.append(r"\clearpage")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(final_output))


def generate_three_long_tables_per_page(model, vars_per_table=50, output_file="variable_table.txt"):
    # Sort the variables by their variable name
    vars_list = sorted(model.getVars(), key=lambda v: v.VarName)

    # Now, total_vars would still be the same
    total_vars = len(vars_list)

    tables = []
    for i in range(0, total_vars, vars_per_table):
        chunk = vars_list[i:i + vars_per_table]

        table = [
            r"\begin{tabular}{|c|c|}",
            r"\hline",
            r"\textbf{Variable} & \textbf{Value} \\",
            r"\hline"
        ]

        for var in chunk:
            name = latex_escape(var.VarName)
            val = var.X
            table.append(f"{name} & {val:.4f} \\\\")
            table.append(r"\hline")

        table.append(r"\end{tabular}")
        tables.append("\n".join(table))

    # Group every 3 tables, then insert a page break
    final_output = []
    for i in range(0, len(tables), 4):
        group = tables[i:i+4]
        final_output.append(r"\noindent")
        for t in group:
            final_output.append(r"{\small")
            final_output.append(t)
            final_output.append(r"}")
            final_output.append(r"\vspace{1em}")
        final_output.append(r"\clearpage")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(final_output))

def get_sorted_vars(model):
    vars_list = model.getVars()
    # Sort the variables by their variable name
    sorted_vars = sorted(vars_list, key=lambda v: v.VarName)

    # Now, total_vars would still be the same
    return len(sorted_vars)
