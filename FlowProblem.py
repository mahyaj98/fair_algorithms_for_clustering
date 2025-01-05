import numpy as np
from collections import defaultdict

from numpy.ma.extras import unique
from scipy.spatial.distance import cdist
from cplex import Cplex
import time

from itertools import permutations

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

    t1 = time.monotonic()
    problem.solve()
    t2 = time.monotonic()
    print("LP solving time = {}".format(t2 - t1))


    flow_res = {
        "status": problem.solution.get_status(),
        "success": problem.solution.get_status_string(),
        "objective": problem.solution.get_objective_value(),
        "assignment": problem.solution.get_values(),
    }

    return flow_res

def fair_flow_lp_solver(df, centers, color_flag, cost_fun_string, t, g_opt, color_lb):
    print("Initializing Cplex model")
    problem = Cplex()

    # Step 2. Declare that this is a minimization problem

    problem.objective.set_sense(problem.objective.sense.minimize)

    # Step 3.   Declare and  add variables to the model. The function
    #           prepare_to_add_variables (points, center) prepares all the
    #           required information for this stage.
    #
    #    objective: a list of coefficients (float) in the linear objective function
    #    lower bounds: a list of floats containing the lower bounds for each variable
    #    upper bounds: a list of floats containing the upper bounds for each variable
    #    variable_name: a list of strings that contains the name of the variables

    print("Starting to add variables...")
    t1 = time.monotonic()
    objective, lower_bounds, upper_bounds, variable_names = prepare_to_add_variables_flow(df, centers, cost_fun_string)
    problem.variables.add(obj=objective,
                          lb=lower_bounds,
                          ub=upper_bounds,
                          names=variable_names)
    t2 = time.monotonic()
    print("Completed. Time for creating and adding variable = {}".format(t2 - t1))

    print("Starting to add constraints...")
    t1 = time.monotonic()
    objects_returned = prepare_to_add_constraints_flow(df, centers, cost_fun_string, color_flag, t, g_opt, color_lb)
    constraints_row, senses, rhs, constraint_names = objects_returned
    problem.linear_constraints.add(lin_expr=constraints_row,
                                   senses=senses,
                                   rhs=rhs,
                                   names=constraint_names)
    t2 = time.monotonic()
    print("Completed. Time for creating and adding constraints = {}".format(t2 - t1))

    # Optional: We can set various parameters to optimize the performance of the lp solver
    # As an example, the following sets barrier method as the lp solving method
    # The other available methods are: auto, primal, dual, sifting, concurrent

    # problem.parameters.lpmethod.set(problem.parameters.lpmethod.values.barrier)

    return problem, objective


def prepare_to_add_variables_flow(df, centers, cost_fun_string):

    num_points = len(df)
    num_centers = len(centers)

    variable_names = ["x_{}_{}".format(j,i) for j in range(num_points) for i in range(num_centers)]


    total_variables = num_points * num_centers
    lower_bounds = [0 for _ in range(total_variables)]
    upper_bounds = [1 for _ in range(total_variables)]


    objective = cost_function_flow(df, centers, cost_fun_string)

    return objective, lower_bounds, upper_bounds, variable_names

def cost_function_flow(df,centers, cost_fun_string):
    all_pair_distance = cdist(df.values,centers,cost_fun_string)
    return all_pair_distance.ravel().tolist()

def cost_function_twoD_flow(df,centers, cost_fun_string):
    all_pair_distance = cdist(df.values,centers,cost_fun_string)
    return all_pair_distance.tolist()


def prepare_to_add_constraints_flow(df, centers, cost_fun_string, color_flag, t, g_opt, color_lb):

    num_points = len(df)
    num_centers = len(centers)


    constraints_row, rhs = constraint_sums_to_one(num_points, num_centers)
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




def constraints_cut_off_opt_flow(num_points, num_centers, distances ,G_OPT):

    constraints = [[["x_{}_{}".format(j, i)],[1 if distances[j][i] > 3 * G_OPT else 0]] for i in range(num_centers) for j in range(num_points)]
    rhs = [0] * (num_points * num_centers)
    return constraints, rhs
def constraint_sums_to_one(num_points, num_centers):

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
