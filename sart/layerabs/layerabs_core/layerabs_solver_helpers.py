from __future__ import annotations

import os
import sys

import gurobipy as gp
from gurobipy import GRB

from sart.layerabs.layerabs_core.layerabs_solver_context import (
    get_solver_time_limit_seconds,
)


class SignConsistencyTerminationCallback:
    def __init__(self):
        self.terminate_solution = False
        self.min_objbst = None
        self.min_objbnd = None
        self.max_objbst = None
        self.max_objbnd = None

    def __call__(self, model, where):
        if where != GRB.Callback.MIP:
            return

        try:
            if model.modelName == "Minimize":
                self.min_objbst = model.cbGet(GRB.Callback.MIP_OBJBST)
                self.min_objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)
            elif model.modelName == "Maximize":
                self.max_objbst = model.cbGet(GRB.Callback.MIP_OBJBST)
                self.max_objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)

            values = [
                self.min_objbst,
                self.min_objbnd,
                self.max_objbst,
                self.max_objbnd,
            ]
            if not all(value is not None for value in values):
                return

            if (
                self.max_objbst > 0
                and self.max_objbnd > 0
                and self.min_objbst > 0
                and self.min_objbnd > 0
            ):
                self.terminate_solution = True
                model.terminate()
            elif (
                self.min_objbst < 0
                and self.min_objbnd < 0
                and self.max_objbst < 0
                and self.max_objbnd < 0
            ):
                self.terminate_solution = True
                model.terminate()
        except Exception as exc:
            print(f"Error in callback: {exc}")


def add_abs_constraints(model, abs_var, expr, M=1000):
    """Add a manual absolute-value encoding when addGenConstrAbs is unsuitable."""
    try:
        var_name = f"{abs_var.VarName}_u"
    except Exception:
        var_name = f"abs_u_{id(abs_var)}"

    u = model.addVar(vtype=GRB.BINARY, name=var_name)
    model.addConstr(expr <= abs_var)
    model.addConstr(-expr <= abs_var)
    model.addConstr(abs_var <= expr + (1 - u) * M)
    model.addConstr(abs_var <= -expr + u * M)
    return u


def build_bound_models(variable_bounds_list):
    """Create quiet min/max Gurobi models and populate the shared variables."""
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")
    model_min = gp.Model("Minimize")
    model_max = gp.Model("Maximize")
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    model_min.setParam("LogToConsole", 0)
    model_max.setParam("LogToConsole", 0)

    solver_time_limit = get_solver_time_limit_seconds()
    if solver_time_limit is not None:
        model_min.setParam("TimeLimit", solver_time_limit)
        model_max.setParam("TimeLimit", solver_time_limit)

    variables_min = {}
    variables_max = {}

    for var_info in variable_bounds_list:
        var_name = var_info[0]
        lb = var_info[1] if var_info[1] is not None else -GRB.INFINITY
        ub = var_info[2] if var_info[2] is not None else GRB.INFINITY

        variables_min[var_name] = model_min.addVar(
            lb=lb,
            ub=ub,
            vtype=GRB.CONTINUOUS,
            name=var_name,
        )
        variables_max[var_name] = model_max.addVar(
            lb=lb,
            ub=ub,
            vtype=GRB.CONTINUOUS,
            name=var_name,
        )

    model_min.update()
    model_max.update()
    return model_min, model_max, variables_min, variables_max


def read_bound_values(model_min, model_max):
    """Read min/max objective values with a bound fallback when not optimal."""
    if model_min.status == GRB.OPTIMAL:
        min_value = model_min.getAttr(GRB.Attr.ObjVal)
    else:
        min_value = model_min.getAttr(GRB.Attr.ObjBound)

    if model_max.status == GRB.OPTIMAL:
        max_value = model_max.getAttr(GRB.Attr.ObjVal)
    else:
        max_value = model_max.getAttr(GRB.Attr.ObjBound)

    min_value = min_value if min_value is not None else float("inf")
    max_value = max_value if max_value is not None else float("-inf")
    return [min_value, max_value]


def add_constraints_to_bound_models(
    constraints,
    model_min,
    model_max,
    variables_min,
    variables_max,
    abs_var_suffix="_Aux",
    manual_abs_constraint_fn=None,
):
    """Add the symbolic constraints to the paired min/max models."""
    for constr in constraints:
        if "Abs(" in constr:
            abs_expr = constr.split("Abs(")[-1].split(")")[0]
            abs_var_name = f"{abs_expr}{abs_var_suffix}"

            if abs_var_name not in variables_min:
                abs_var_min = model_min.addVar(
                    name=abs_var_name,
                    lb=0,
                    ub=GRB.INFINITY,
                    vtype=GRB.CONTINUOUS,
                )
                abs_var_max = model_max.addVar(
                    name=abs_var_name,
                    lb=0,
                    ub=GRB.INFINITY,
                    vtype=GRB.CONTINUOUS,
                )

                variables_min[abs_var_name] = abs_var_min
                variables_max[abs_var_name] = abs_var_max

                evaluated_abs_expr_min = eval(abs_expr, {}, variables_min)
                evaluated_abs_expr_max = eval(abs_expr, {}, variables_max)

                if manual_abs_constraint_fn is None:
                    model_min.addGenConstrAbs(abs_var_min, evaluated_abs_expr_min)
                    model_max.addGenConstrAbs(abs_var_max, evaluated_abs_expr_max)
                else:
                    manual_abs_constraint_fn(
                        model_min,
                        abs_var_min,
                        evaluated_abs_expr_min,
                    )
                    manual_abs_constraint_fn(
                        model_max,
                        abs_var_max,
                        evaluated_abs_expr_max,
                    )

            constr = constr.replace(f"Abs({abs_expr})", abs_var_name)

        model_min.addConstr(eval(constr, {}, variables_min))
        model_max.addConstr(eval(constr, {}, variables_max))


def optimize_bound_models(
    model_min,
    model_max,
    variables_min,
    variables_max,
    objective,
    callback=None,
    should_terminate=None,
):
    """Set objectives, run the paired solves, and return the final bounds."""
    model_min.setObjective(eval(objective, {}, variables_min), GRB.MINIMIZE)
    model_max.setObjective(eval(objective, {}, variables_max), GRB.MAXIMIZE)

    if callback is None:
        model_min.optimize()
        model_max.optimize()
    else:
        model_min.optimize(callback)
        model_max.optimize(callback)

    if should_terminate is not None and should_terminate():
        model_min.terminate()
        model_max.terminate()

    return read_bound_values(model_min, model_max)


def optimize_with_bounds(constraints, objective, variable_bounds_list):
    callback = SignConsistencyTerminationCallback()
    model_min, model_max, variables_min, variables_max = build_bound_models(
        variable_bounds_list
    )
    add_constraints_to_bound_models(
        constraints,
        model_min,
        model_max,
        variables_min,
        variables_max,
        abs_var_suffix="_Aux",
    )
    return optimize_bound_models(
        model_min,
        model_max,
        variables_min,
        variables_max,
        objective,
        callback=callback,
        should_terminate=lambda: callback.terminate_solution,
    )
