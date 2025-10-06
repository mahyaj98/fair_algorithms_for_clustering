import numpy as np
from numpy.ma.extras import unique
from scipy.spatial.distance import cdist
from FlowProblem_gurobi import construct_flow_lp
from FlowInteger_gurobi import construct_flowint_lp
from itertools import permutations
import gurobipy as gp
from gurobipy import GRB



def fair_partial_assignment(df, centers, color_flag, attributes, t, g_opt):

    cost_fun_string = 'euclidean'
    problem, objective = fair_partial_assignment_lp_solver(df, centers, color_flag, cost_fun_string,t, g_opt)
    attr = list(color_flag.keys())[0]
    problem.optimize()

    res = {
        "success": problem.Status,
        "objective": problem.ObjVal if problem.Status == 2 else -1,
        "assignment": [var.X for var in problem.getVars()] if problem.Status == 2 else [0 for _ in problem.getVars()],
    }


    if res["success"] == 2:
        flow_res = construct_flow_lp(df,centers,color_flag ,attributes,res,t, g_opt)
        if flow_res["success"] == 2:

            flow_res["partial_assignment"] = res["assignment"]
            flow_res["partial_objective"] = res["objective"]

            flow_assignment = np.array(flow_res["assignment"]).reshape(len(df), len(centers)).tolist()
            assignment, unassigned, color_per_centre =  unassign_violations(flow_assignment, color_flag[attr], t, centers)
            need_to_reassign = unassigned.__len__()
            while unassigned.__len__() > 0:
                balance_f = True
                for unassigned_pair in unassigned:
                    reassigned_f, assignment, color_per_centre = simple_reassign(assignment, unassigned_pair, centers, color_per_centre, t)
                    if reassigned_f:
                        unassigned.remove(unassigned_pair)
                        balance_f = False
                        break
                if balance_f:
                    assignment, color_per_centre = balance_min(centers, color_per_centre, assignment, color_flag[attr])

            flow_res["assignment"] = assignment
            distances = cost_function( df, centers, cost_fun_string )
            assignment_1d = np.array( flow_assignment ).ravel()
            bench_cost = np.inner( assignment_1d, distances )
            flow_res["fair_cost"] = bench_cost
            intflow_res = construct_flowint_lp(df, centers, color_flag, color_per_centre)
            if intflow_res["success"] == 2:
                flow_res ["fair_cost"] = intflow_res["objective"]
                flow_res ["assignment"] = intflow_res["assignment"]
                assignment = np.array(intflow_res["assignment"]).reshape(len(df), len(centers)).tolist()
                color_list = unique( color_flag[attr] )
                color_per_centre = {}
                for i in range( len( centers ) ):
                    color_per_centre [i] = {}
                    for color in range( len( color_list ) ):
                        color_per_centre [i] [color] = sum(
                            [assignment [j] [i] if color_flag[attr] [j] == color else 0 for j in range( len( assignment ) )] )

                flow_res ["color_per_centre"] = color_per_centre
                return flow_res, need_to_reassign
            else:
                return flow_res, need_to_reassign
        else: return flow_res, 0
    else:
        return res, 0

def is_color_and_centre(client, color, centre, assignment, color_flag):
    if color_flag[client] == color and assignment[client][centre] == 1:
        return True
    return False

def balance_min(centres, color_per_centres, assignment, color_flag):

    color_list = color_per_centres[0].keys()
    for color in color_list:
        for cent in range(len(centres)):
            min_0 = min(color_per_centres[0].values())
            min_cent = min(color_per_centres[cent].values())
            if color_per_centres[cent][color] > min_cent and color_per_centres[0][color] == min_0:
                for j in range(len(assignment)):
                    if is_color_and_centre(j, color, cent, assignment, color_flag):
                        assignment[j][cent] = 0
                        assignment[j][0] = 1
                        color_per_centres[cent][color] -= 1
                        color_per_centres[0][color] += 1
                        break
    return assignment, color_per_centres

def is_fair_centre(color_in_centre, t):
    for (c1,c2) in permutations(color_in_centre.keys(),2):
        if c1 != c2 and color_in_centre[c1] > t * color_in_centre[c2]:
            return False, c1, c2, color_in_centre[c1] - t * color_in_centre[c2]

    return True, None, None, None

def is_there_space(color_per_centre, t, in_color):
    color_list = color_per_centre.keys()
    for color in color_list:
        if color_per_centre[in_color] +1 > color_per_centre[color]*t:
            return False
    return True

def unassign_from_color(assignment, color, diff, color_flag, centre, color_per_centre):
    counter = 0
    unassigned_pairs = []
    tmp_assignment = assignment
    for j in range(len(assignment)):
        if color_flag[j] == color and assignment[j][centre] == 1:
            counter += 1
            tmp_assignment[j][centre] = 0
            color_per_centre[color]-=1
            unassigned_pairs.append((j,color))
            if counter >= diff:
                break
    return unassigned_pairs, tmp_assignment,color_per_centre

def unassign_violations(assignment, color_flag, t, centres):
    unassigned = []
    color_list = unique(color_flag)
    color_per_centre = {}
    for i in range(len(centres)):
        color_per_centre[i] = {}
        for color in range(len(color_list)):
            color_per_centre[i][color] = sum([assignment[j][i] if color_flag[j] == color else 0 for j in range(len(assignment))])

    for i in range(len(centres)):
        is_fair, color1, color2, diff = is_fair_centre(color_per_centre[i], t)
        while not is_fair:
            unassigned_pairs, tmp_assignment, color_per_centre[i] = unassign_from_color(assignment, color1, diff, color_flag, i, color_per_centre[i])
            assignment = tmp_assignment
            unassigned+= unassigned_pairs
            is_fair, color1, color2, diff = is_fair_centre(color_per_centre[i], t)
    return assignment, unassigned, color_per_centre

def simple_reassign(assignment, unassigned_pair, centres, color_per_centre, t):

    j, color = unassigned_pair[0], unassigned_pair[1]

    for i in range(len(centres)):
        if is_there_space(color_per_centre[i], t, color):
            assignment[j][i] = 1
            color_per_centre[i][color] += 1
            return True, assignment, color_per_centre

    return False, assignment, color_per_centre

def fair_partial_assignment_lp_solver(df, centers, color_flag, cost_fun_string, t, g_opt):



    problem = gp.Model("mip1")





    lower_bounds, upper_bounds, variable_names = prepare_to_add_variables(df, centers)

    varname_to_var = {}
    for i in range(len(variable_names)):
        varname_to_var[variable_names[i]] = problem.addVar(lb=lower_bounds[i], ub=upper_bounds[i], name=variable_names[i])






    objects_returned = prepare_to_add_constraints(df, centers, cost_fun_string,color_flag, t, g_opt)
    constraints_row, senses, rhs, constraint_names = objects_returned
    for i in range(len(constraints_row)):
        if senses[i] == "E":
            problem.addConstr( sum(
                [varname_to_var[name] * coef for (name, coef) in
                 zip( constraints_row [i] [0], constraints_row [i] [1] )] ) == rhs [i]
                               , name=constraint_names [i] )
        elif senses[i] == "G":
            problem.addConstr(sum(
                [varname_to_var[name] * coef for (name, coef) in
                 zip(constraints_row[i][0], constraints_row[i][1])]) >= rhs[i]
                              , name=constraint_names[i])
        else:
            problem.addConstr(sum(
                [varname_to_var[name] * coef for (name, coef) in
                 zip(constraints_row[i][0], constraints_row[i][1])]) <= rhs[i]
                              , name=constraint_names[i])



    objective = cost_function(df, centers, cost_fun_string)

    problem.setObjective(sum(
                [varname_to_var[name] * distance for (name, distance) in
                 zip(variable_names, objective)]),
        GRB.MINIMIZE
    )



    return problem, objective

def prepare_to_add_variables(df, centers):

    num_points = len(df)
    num_centers = len(centers)

    variable_names = ["x_{}_{}".format(j,i) for j in range(num_points) for i in range(num_centers)]


    total_variables = num_points * num_centers
    lower_bounds = [0 for _ in range(total_variables)]
    upper_bounds = [1 for _ in range(total_variables)]



    return lower_bounds, upper_bounds, variable_names

def cost_function(df,centers, cost_fun_string):
    all_pair_distance = cdist(df.values,centers,cost_fun_string)
    return all_pair_distance.ravel().tolist()

def cost_function_twoD(df,centers, cost_fun_string):
    all_pair_distance = cdist(df.values,centers,cost_fun_string)
    return all_pair_distance.tolist()

def prepare_to_add_constraints(df, centers, cost_fun_string, color_flag, t, g_opt):

    num_points = len(df)
    num_centers = len(centers)


    constraints_row, rhs = constraint_sums_to_one(num_points, num_centers)
    sum_const_len = len(rhs)

    distances = cost_function_twoD(df, centers, cost_fun_string)
    cutoff_constraints, cutoff_rhs = constraints_cut_off_opt(num_points, num_centers, distances, g_opt)
    cutoff_const_len = len(cutoff_rhs)
    constraints_row.extend(cutoff_constraints)
    rhs.extend(cutoff_rhs)

    for var in color_flag:
        color_constraint, color_rhs = constraint_color(num_points, num_centers, color_flag[var],t)
        constraints_row.extend(color_constraint)
        rhs.extend(color_rhs)

    senses = (["E" for _ in range(sum_const_len)] +
              ["E" for _ in range(cutoff_const_len)]
              +["G" for _ in range(len(rhs) - sum_const_len-cutoff_const_len)])

    constraint_names = ["c_{}".format(i) for i in range(len(rhs))]

    return constraints_row, senses, rhs, constraint_names

def constraints_cut_off_opt(num_points, num_centers, distances ,g_opt):

    constraints = [[["x_{}_{}".format(j, i)],[1 if distances[j][i] > g_opt else 0]] for i in range(num_centers) for j in range(num_points)]
    rhs = [0] * (num_points * num_centers)
    return constraints, rhs

def constraint_sums_to_one(num_points, num_centers):

    constraints = [[["x_{}_{}".format(j, i) for i in range(num_centers)], [1] * num_centers] for j in range(num_points)]
    rhs = [1] * num_points
    return constraints, rhs

def constraint_color(num_points, num_centers, color_flag,t):

    color_list = unique(color_flag)
    t = int(t)
    t_constraints = [[["x_{}_{}".format(j, i) for j in range(num_points)],
                         [-1 if color_flag[j] == color1 else t if color_flag[j] == color2 else 0 for j in range(num_points)]]
                        for i in range(num_centers) for (color1, color2) in permutations(color_list,2)]

    constraints = t_constraints
    number_of_constraints = len(t_constraints)
    rhs = [0] * number_of_constraints
    return constraints, rhs
