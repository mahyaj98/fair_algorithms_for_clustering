import numpy as np
from collections import defaultdict
from numpy.ma.extras import unique
from scipy.spatial.distance import cdist
import gurobipy as gp
from gurobipy import GRB

def construct_flowint_lp(df, centres, color_flag, color_per_centre):


    cost_fun_string = 'euclidean'
    problem, objective = fair_flowint_lp_solver(df, centres, color_flag, cost_fun_string, color_per_centre)

    problem.optimize()

    if problem.Status != 2:
        flow_res = {
            "success": problem.Status,
            "objective": -1,
            "assignment": [0 for var in problem.getVars()],
        }
    else:
        flow_res = {
            "success": problem.Status,
            "objective": problem.ObjVal,
            "assignment": [var.X for var in problem.getVars()],
        }

    return flow_res

def fair_flowint_lp_solver(df, centers, color_flag, cost_fun_string, color_per_centre):
    problem = gp.Model("mip2")

    lower_bounds, upper_bounds, variable_names = prepare_to_add_variables_flowint( df, centers )

    varname_to_var = {}
    for i in range( len( variable_names ) ):
        varname_to_var [variable_names [i]] = problem.addVar( lb=lower_bounds [i], ub=upper_bounds [i],
                                                              name=variable_names [i] )

    objects_returned = prepare_to_add_constraints_flowint( df, centers, cost_fun_string, color_flag, color_per_centre)
    constraints_row, senses, rhs, constraint_names = objects_returned
    for i in range( len( constraints_row ) ):
        if senses [i] == "E":
            problem.addConstr( sum(
                [varname_to_var [name] * coef for (name, coef) in
                 zip( constraints_row [i] [0], constraints_row [i] [1] )] ) == rhs [i]
                               , name=constraint_names [i] )
        elif senses [i] == "G":
            problem.addConstr( sum(
                [varname_to_var [name] * coef for (name, coef) in
                 zip( constraints_row [i] [0], constraints_row [i] [1] )] ) >= rhs [i]
                               , name=constraint_names [i] )
        else:
            problem.addConstr( sum(
                [varname_to_var [name] * coef for (name, coef) in
                 zip( constraints_row [i] [0], constraints_row [i] [1] )] ) <= rhs [i]
                               , name=constraint_names [i] )

    objective = cost_function_flowint( df, centers, cost_fun_string )

    problem.setObjective( sum(
        [varname_to_var [name] * distance for (name, distance) in
         zip( variable_names, objective )] ),
        GRB.MINIMIZE
    )

    return problem, objective

def prepare_to_add_variables_flowint(df, centers):

    num_points = len(df)
    num_centers = len(centers)

    variable_names = ["x_{}_{}".format(j,i) for j in range(num_points) for i in range(num_centers)]


    total_variables = num_points * num_centers
    lower_bounds = [0 for _ in range(total_variables)]
    upper_bounds = [1 for _ in range(total_variables)]



    return lower_bounds, upper_bounds, variable_names

def cost_function_flowint(df,centers, cost_fun_string):
    all_pair_distance = cdist(df.values,centers,cost_fun_string)
    return all_pair_distance.ravel().tolist()

def cost_function_twoD_flowint(df,centers, cost_fun_string):
    all_pair_distance = cdist(df.values,centers,cost_fun_string)
    return all_pair_distance.tolist()

def prepare_to_add_constraints_flowint(df, centers, cost_fun_string, color_flag, color_per_centre):

    num_points = len(df)
    num_centers = len(centers)


    constraints_row, rhs = constraint_sums_to_one_flowint(num_points, num_centers)
    sum_const_len = len(rhs)


    for var in color_flag:
        color_constraint, color_rhs = constraint_color_flowint(num_points, num_centers, color_flag[var],color_per_centre)
        constraints_row.extend(color_constraint)
        rhs.extend(color_rhs)

    senses = (["E" for _ in range(sum_const_len)] +
              ["E" for _ in range(len(rhs) - sum_const_len)])

    constraint_names = ["c_{}".format(i) for i in range(len(rhs))]

    return constraints_row, senses, rhs, constraint_names

def constraint_sums_to_one_flowint(num_points, num_centers):

    constraints = [[["x_{}_{}".format(j, i) for i in range(num_centers)], [1] * num_centers] for j in range(num_points)]
    rhs = [1] * num_points
    return constraints, rhs

def constraint_color_flowint(num_points, num_centers, color_flag, color_per_centre):

    color_list = unique(color_flag)
    constraints = [[["x_{}_{}".format(j, i) for j in range(num_points)],
                         [1 if color_flag[j] == color else 0 for j in range(num_points)]]
                        for i in range(num_centers) for color in color_list]

    rhs = [np.floor(color_per_centre[i][color]) for i in range(num_centers) for color in color_list]


    return constraints, rhs
