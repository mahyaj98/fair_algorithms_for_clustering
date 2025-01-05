import numpy as np
from numpy.ma.extras import unique
from scipy.spatial.distance import cdist
from cplex import Cplex
import time
from FlowProblem import construct_flow_lp
from itertools import permutations
from collections import defaultdict



def fair_partial_assignment(df, centers, color_flag, attributes, t, g_opt):

    cost_fun_string = 'euclidean'
    problem, objective = fair_partial_assignment_lp_solver(df, centers, color_flag, cost_fun_string,t, g_opt)

    t1 = time.monotonic()
    problem.solve()
    t2 = time.monotonic()
    print("LP solving time = {}".format(t2-t1))


    res = {
        "status": problem.solution.get_status(),
        "success": problem.solution.get_status_string(),
        "objective": problem.solution.get_objective_value(),
        "assignment": problem.solution.get_values(),
    }


    if res["success"] == "optimal":
        flow_res = construct_flow_lp(df,centers,color_flag ,attributes,res,t, g_opt)

        flow_res["partial_assignment"] = res["assignment"]
        flow_res["partial_objective"] = res["objective"]

        flow_assignment = np.array(flow_res["assignment"]).reshape(len(df), len(centers)).tolist()
        assignment, unassigned, color_per_centre =  unassign_violations(flow_assignment, color_flag['marital'], t, centers)
        while unassigned.__len__() > 0:
            assignment, unassigned, color_per_centre = type_one_reassign(assignment, unassigned, centers,
                                                                         color_per_centre, t)
            assignment, color_per_centre = balance_min(centers, color_per_centre, assignment, color_flag['marital'])

        flow_res["assignment"] = assignment


        return flow_res
    else:
        return res
def is_color_and_centre(client, color, centre, assignment, color_flag):
    if color_flag[client] == color and assignment[client][centre] == 1:
        return True
    return False

def balance_min(centres, color_per_centres, assignment, color_flag):

    color_list = color_per_centres[0].keys()
    for (cent1,cent2) in permutations(range(len(centres)),2):
        min_1 = min(color_per_centres[cent1].values())
        min_2 = min(color_per_centres[cent2].values())
        for color in color_list:
            if color_per_centres[cent1][color] > min_1 and color_per_centres[cent2][color] == min_2:
                for j in range(len(assignment)):
                    if is_color_and_centre(j, color, cent1, assignment, color_flag):
                        assignment[j][cent1] = 0
                        assignment[j][cent2] = 1
                        color_per_centres[cent1][color] -= 1
                        color_per_centres[cent2][color] += 1
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

def type_one_reassign(assignment, unassigned_pairs, centres, color_per_centre, t):
    tmp_unassigned_pairs = unassigned_pairs.copy()

    for (j,color) in unassigned_pairs:
        for i in range(len(centres)):
            if is_there_space(color_per_centre[i], t, color):
                assignment[j][i] = 1
                color_per_centre[i][color] += 1
                tmp_unassigned_pairs.remove((j,color))
                break


    return assignment, tmp_unassigned_pairs, color_per_centre




def fair_partial_assignment_lp_solver(df, centers, color_flag, cost_fun_string, t, g_opt):



    print("Initializing Cplex model")
    problem = Cplex()


    problem.objective.set_sense(problem.objective.sense.minimize)



    print("Starting to add variables...")
    t1 = time.monotonic()
    objective, lower_bounds, upper_bounds, variable_names = prepare_to_add_variables(df, centers, cost_fun_string)
    problem.variables.add(obj=objective,
                          lb=lower_bounds,
                          ub=upper_bounds,
                          names=variable_names)
    t2 = time.monotonic()
    print("Completed. Time for creating and adding variable = {}".format(t2-t1))



    print("Starting to add constraints...")
    t1 = time.monotonic()
    objects_returned = prepare_to_add_constraints(df, centers, cost_fun_string,color_flag, t, g_opt)
    constraints_row, senses, rhs, constraint_names = objects_returned
    problem.linear_constraints.add(lin_expr=constraints_row,
                                   senses=senses,
                                   rhs=rhs,
                                   names=constraint_names)
    t2 = time.monotonic()
    print("Completed. Time for creating and adding constraints = {}".format(t2-t1))



    return problem, objective




def prepare_to_add_variables(df, centers, cost_fun_string):

    num_points = len(df)
    num_centers = len(centers)

    variable_names = ["x_{}_{}".format(j,i) for j in range(num_points) for i in range(num_centers)]


    total_variables = num_points * num_centers
    lower_bounds = [0 for _ in range(total_variables)]
    upper_bounds = [1 for _ in range(total_variables)]


    objective = cost_function(df, centers, cost_fun_string)

    return objective, lower_bounds, upper_bounds, variable_names




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




def constraints_cut_off_opt(num_points, num_centers, distances ,G_OPT):

    constraints = [[["x_{}_{}".format(j, i)],[1 if distances[j][i] > 3 * G_OPT else 0]] for i in range(num_centers) for j in range(num_points)]
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
