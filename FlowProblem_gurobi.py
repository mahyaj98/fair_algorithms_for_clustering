import numpy as np
from collections import defaultdict
from numpy.ma.extras import unique
from scipy.spatial.distance import cdist
import gurobipy as gp
from gurobipy import GRB

def construct_flow_lp(df, centres, color_flag, atrributes, res, t, g_opt):
    color_lb = {}
    assignment_res = np.array(res['assignment'])
    assignment_res = assignment_res.reshape(len(df), len(centres)).tolist()
    for var in atrributes:
        color_lb[var] = defaultdict(int)
        for color in atrributes[var]:
            color_lb[var][color] = defaultdict(int)
            for i in range(len(centres)):
                color_lb[var][color][i] = sum(assignment_res[index][i] for index in atrributes[var][color])


    cost_fun_string = 'euclidean'
    problem, objective = fair_flow_lp_solver(df, centres, color_flag, cost_fun_string, t, g_opt, color_lb)

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

def fair_flow_lp_solver(df, centers, color_flag, cost_fun_string, t, g_opt, color_lb):
    problem = gp.Model("mip2")

    lower_bounds, upper_bounds, variable_names = prepare_to_add_variables_flow( df, centers )

    varname_to_var = {}
    for i in range( len( variable_names ) ):
        varname_to_var [variable_names [i]] = problem.addVar( lb=lower_bounds [i], ub=upper_bounds [i],
                                                              name=variable_names [i] )

    objects_returned = prepare_to_add_constraints_flow( df, centers, cost_fun_string, color_flag, t, g_opt , color_lb)
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

    objective = cost_function_flow( df, centers, cost_fun_string )

    problem.setObjective( sum(
        [varname_to_var [name] * distance for (name, distance) in
         zip( variable_names, objective )] ),
        GRB.MINIMIZE
    )

    return problem, objective

def prepare_to_add_variables_flow(df, centers):

    num_points = len(df)
    num_centers = len(centers)

    variable_names = ["x_{}_{}".format(j,i) for j in range(num_points) for i in range(num_centers)]


    total_variables = num_points * num_centers
    lower_bounds = [0 for _ in range(total_variables)]
    upper_bounds = [1 for _ in range(total_variables)]



    return lower_bounds, upper_bounds, variable_names

def cost_function_flow(df,centers, cost_fun_string):
    all_pair_distance = cdist(df.values,centers,cost_fun_string)
    return all_pair_distance.ravel().tolist()

def cost_function_twoD_flow(df,centers, cost_fun_string):
    all_pair_distance = cdist(df.values,centers,cost_fun_string)
    return all_pair_distance.tolist()

def prepare_to_add_constraints_flow(df, centers, cost_fun_string, color_flag, t, g_opt, color_lb):

    num_points = len(df)
    num_centers = len(centers)


    constraints_row, rhs = constraint_sums_to_one_flow(num_points, num_centers)
    sum_const_len = len(rhs)

    distances = cost_function_twoD_flow(df, centers, cost_fun_string)
    cutoff_constraints, cutoff_rhs = constraints_cut_off_opt_flow(num_points, num_centers, distances, g_opt)
    cutoff_const_len = len(cutoff_rhs)
    constraints_row.extend(cutoff_constraints)
    rhs.extend(cutoff_rhs)


    for var in color_flag:
        color_constraint, color_rhs = constraint_color_flow(num_points, num_centers, color_flag[var],t, color_lb[var])
        constraints_row.extend(color_constraint)
        rhs.extend(color_rhs)

    senses = (["E" for _ in range(sum_const_len)] +
              ["E" for _ in range(cutoff_const_len)]
              +["L" for _ in range(len(rhs) - sum_const_len-cutoff_const_len)])

    constraint_names = ["c_{}".format(i) for i in range(len(rhs))]

    return constraints_row, senses, rhs, constraint_names

def constraints_cut_off_opt_flow(num_points, num_centers, distances ,g_opt):

    constraints = [[["x_{}_{}".format(j, i)],[1 if distances[j][i] > 3 * g_opt else 0]] for i in range(num_centers) for j in range(num_points)]
    rhs = [0] * (num_points * num_centers)
    return constraints, rhs

def constraint_sums_to_one_flow(num_points, num_centers):

    constraints = [[["x_{}_{}".format(j, i) for i in range(num_centers)], [1] * num_centers] for j in range(num_points)]
    rhs = [1] * num_points
    return constraints, rhs

def constraint_color_flow(num_points, num_centers, color_flag,t, color_lb):

    color_list = unique(color_flag)
    t = int(t)
    lhs_constraints = [[["x_{}_{}".format(j, i) for j in range(num_points)],
                         [-1 if color_flag[j] == color else 0 for j in range(num_points)]]
                        for i in range(num_centers) for color in color_list]

    rhs_1 = [-np.floor(color_lb[color][i]) for i in range(num_centers) for color in color_list]

    rhs_constraints = [[["x_{}_{}".format(j, i) for j in range(num_points)],
                        [1 if color_flag[j] == color else 0 for j in range(num_points)]]
                       for i in range(num_centers) for color in color_list]

    rhs_2 = [np.ceil(t * color_lb[color][i]) for i in range(num_centers) for color in color_list]

    constraints = lhs_constraints + rhs_constraints
    rhs = rhs_1 + rhs_2
    return constraints, rhs
