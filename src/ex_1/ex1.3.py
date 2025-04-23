
import argparse

import gurobipy as gp
from gurobipy import GRB
from src.util.utils import generate_latex_table, generate_three_tables_per_page,generate_three_long_tables_per_page


def build_model(model: gp.Model, n: int, k: int):
    p = model.addVars(
        n,
        name="p",
        vtype=GRB.INTEGER,
    )

    d = model.addVars(
        [(i,j,g) for i in range(n) for j in range(n) for g in range(2)],
        name="d",
        vtype=GRB.BINARY,
    )
    w = model.addVars(
        [(i,j,g) for i in range(n) for j in range(n) for g in range(2)],
        name="w",
        vtype=GRB.BINARY,
    )

    z = model.addVars(
        n,
        name="z",
        vtype=GRB.BINARY,
    )
    r = model.addVars(
        n,
        name="r",
        vtype=GRB.BINARY,
    )
    mu = model.addVar(
        name="µ",
        vtype=GRB.INTEGER,
    )

    m = 6 * (n-1) # maximum number of point a team can achieve in theory (winning every game)

    # constraints
    model.addConstrs((d[i,j,g] == d[j,i,g] for i in range(n) for j in range(n) for g in range(2)),
                     name="If game (i,j) ends in a draw, both teams receive the draw")
    model.addConstrs((w[i,j,g] + w[j,i,g] + d[i,j,g] == 1 for i in range(n) for j in range(n) for g in range(2) if j != i),
                     name="In a game (i,j), there can only be one winner or a draw")
    model.addConstrs((w[i, i, g] + w[i, i, g] + d[i, i, g] == 0 for i in range(n) for g in range(2)),
                     name="In a game (i,i), no self loops")
    model.addConstrs((gp.quicksum(3 * w[i,j,g] + d[i,j,g] for j in range(n) for g in range(2) if j != i) == p[i] for i in range(n)),
                     name="Sum of points received by all games for team i")

    # r constraints
    model.addConstrs((p[i] <= mu + m * (1-r[i]) for i in range(n)),
                     name="If team i has <= points than µ, it must get relegated")
    model.addConstrs((p[i] >= mu - m * r[i] for i in range(n)),
                     name="If team i has > points than µ, it must not get relegated")
    model.addConstr(gp.quicksum(r[i] for i in range(n)) == k,
                    name="There must be exactly k teams that get relegated")

    # z constraints
    model.addConstrs((mu >= p[i] - m * (1 - z[i]) for i in range(n)),
                     name="z_i must be 1 if p_i is <= to mu")
    model.addConstrs((mu <= p[i] + m * (1 - z[i]) for i in range(n)),
                     name="z_i must be 1 if p_i is >= to mu")
    # gurobi alternative to the two constraints above (indicator constraint feature)
    # model.addConstrs(
    #     (z[i] == 1) >> (p[i] == mu)
    #     for i in range(n)
    # )
    model.addConstrs((z[i] <= r[i] for i in range(n)),
                     name="The team with points equivalent to mu must be one of the relegated ones")
    model.addConstr(gp.quicksum(z[i] for i in range(n)) >= 1,
                    name="µ must be equal to the points of at least one team")

    # debug constraints for verification purposes
    # D1. Checks if the distribution of points and game results match
    # model.addConstr(
    #     gp.quicksum(p[i] for i in range(n)) ==
    #     gp.quicksum(3 * w[i, j, g] + d[i, j, g] for i in range(n) for j in range(n) for g in range(2) if i != j),
    #     name="TotalPointsFromGames"
    # )
    # D2. Sets the points of team 0 to a specific value to check if that is indeed a non-relegation guarantee
    # model.addConstr(p[0] == 58,name="Test")


    # we want to push mu as high as possible and add 1 point to ensure non-relegation
    model.setObjective(mu + 1, GRB.MAXIMIZE)




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

        latex_table = generate_three_long_tables_per_page(model)

    model.close()
