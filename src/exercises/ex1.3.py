import argparse

import gurobipy as gp
from gurobipy import GRB


def build_model(model: gp.Model, n: int, k: int):
    # put your model building code here
    #
    # x = model.addVars(...)
    #
    # if you want to access your variables outside this function, you can use
    # model._x = x
    # to save a reference in the model itself
    #
    # model.addConstrs(...)
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", default=18)
    parser.add_argument("--k", default=3)
    args = parser.parse_args()

    model = gp.Model("ex1.3")
    build_model(model, args.n, args.k)

    model.update()
    model.optimize()

    if model.SolCount > 0:
        print(f"obj. value = {model.ObjVal}")
        for v in model.getVars():
            print(f"{v.VarName} = {v.X}")

    model.close()
