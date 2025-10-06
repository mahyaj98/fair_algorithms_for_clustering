import configparser
import json
import time
from collections import defaultdict
from functools import partial
import os
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import pairwise_distances

from pyarrow.dataset import dataset
from sympy import ceiling
import pandas as pd
import matplotlib.pyplot as plt

# from cplex_fair_assignment_lp_solver import fair_partial_assignment

from gurobi_fair_assignment_lp_solver import fair_partial_assignment

from util.clusteringutil import (clean_data, read_data, scale_data,
                                 subsample_data, take_by_key,
                                 vanilla_clustering, write_fairness_trial)
from util.configutil import read_list


# This function takes a dataset and performs a fair clustering on it.
# Arguments:
#   dataset (str) : dataset to use
#   config_file (str) : config file to use (will be read by ConfigParser)
#   data_dir (str) : path to write output
#   num_clusters (int) : number of clusters to use
#   deltas (list[float]) : delta to use to tune alpha, beta for each color
#   max_points (int ; default = 0) : if the number of points in the dataset 
#       exceeds this number, the dataset will be subsampled to this amount.
# Output:
#   None (Writes to file in `data_dir`)
def cost_function_euc(df):
    all_pair_distance = pairwise_distances(df.values)
    return all_pair_distance.ravel().tolist()

def fair_clustering(dataset: str, config_file: str, data_dir: str, num_clusters: int, max_points: int, trial_number: int) -> None:
    config = configparser.ConfigParser(converters={'list': read_list})
    config.read(config_file)

    # Read data in from a given csv_file found in config
    # df (pd.DataFrame) : holds the data
    df = read_data(config, dataset)

    # Subsample data if needed
    if max_points and len(df) > max_points:
       df = subsample_data(df, max_points)

    # Clean the data (bucketize text data)
    df, _ = clean_data(df, config, dataset)

    # variable_of_interest (list[str]) : variables that we would like to collect statistics for
    variable_of_interest = config[dataset].getlist("variable_of_interest")

    # Assign each data point to a color, based on config file
    # attributes (dict[str -> defaultdict[int -> list[int]]]) : holds indices of points for each color class
    # color_flag (dict[str -> list[int]]) : holds map from point to color class it belongs to (reverse of `attributes`)
    attributes, color_flag = {}, {}
    for variable in variable_of_interest:
        colors = defaultdict(list)
        this_color_flag = [0] * len(df)
        
        condition_str = variable + "_conditions"
        bucket_conditions = config[dataset].getlist(condition_str)

        # For each row, if the row passes the bucket condition,
        # then the row is added to that color class
        for i, row in df.iterrows():
            for bucket_idx, bucket in enumerate(bucket_conditions):
                if eval(bucket)(row[variable]):
                    colors[bucket_idx].append(i)
                    this_color_flag[i] = bucket_idx

        attributes[variable] = colors
        color_flag[variable] = this_color_flag

    # representation (dict[str -> dict[int -> float]]) : representation of each color compared to the whole dataset
    representation = {}

    for var, bucket_dict in attributes.items():
        representation[var] = {k : (len(bucket_dict[k]) / len(df)) for k in bucket_dict.keys()}

    attr = list(attributes.keys())[0]
    t = ceiling(max([len(attributes[attr][color]) for color in attributes[attr].keys()])/min([len(attributes[attr][color]) for color in attributes[attr].keys()]))
    print( "t= " + str( t ) )

    selected_columns = config[dataset].getlist("columns")
    df = df[[col for col in selected_columns]]

    # Scale data if desired
    scaling = config["DEFAULT"].getboolean("scaling")
    if scaling:
        df = scale_data(df)

    # Cluster the data -- using the objective specified by clustering_method
    clustering_method = config["DEFAULT"]["clustering_method"]

    t1 = time.monotonic()
    initial_score, pred, cluster_centers = vanilla_clustering(df, num_clusters, clustering_method)
    t2 = time.monotonic()
    cluster_time = t2 - t1
    print("Clustering time: {}".format(cluster_time))



    ### Calculate fairness statistics
    # fairness ( dict[str -> defaultdict[int-> defaultdict[int -> int]]] )
    # fairness : is used to hold how much of each color belongs to each cluster
    fairness = {}
    # For each point in the dataset, assign it to the cluster and color it belongs too
    for attr, colors in attributes.items():
        fairness[attr] = defaultdict(partial(defaultdict, int))
        for i, row in enumerate(df.iterrows()):
            cluster = pred[i]
            for color in colors:
                if i in colors[color]:
                    fairness[attr][cluster][color] += 1
                    continue

    # sizes (list[int]) : sizes of clusters
    sizes = [0 for _ in range(num_clusters)]
    for p in pred:
        sizes[p] += 1

    # ratios (dict[str -> dict[int -> list[float]]]): Ratios for colors in a cluster
    ratios = {}
    for attr, colors in attributes.items():
        attr_ratio = {}
        for cluster in range(num_clusters):
            attr_ratio[cluster] = [fairness[attr][cluster][color] / sizes[cluster]
                            for color in sorted(colors.keys())]
        ratios[attr] = attr_ratio

    color_per_centre_vanilla = {}
    for attr, colors in attributes.items():
        attr_count = {}
        for cluster in range(num_clusters):
            attr_count[cluster] = [fairness[attr][cluster][color]
                                   for color in sorted(colors.keys())]
        color_per_centre_vanilla[attr] = attr_count


    # dataset_ratio : Ratios for colors in the dataset
    dataset_ratio = {}
    for attr, color_dict in attributes.items():
        dataset_ratio[attr] = {int(color) : len(points_in_color) / len(df) 
                            for color, points_in_color in color_dict.items()}

    # fairness_vars (list[str]) : Variables to perform fairness balancing on
    fairness_vars = config[dataset].getlist("fairness_variable")

    fp_color_flag = take_by_key(color_flag, fairness_vars)
    fp_attributes = take_by_key(attributes, fairness_vars)

    distances = cost_function_euc(df)
    max_distance = max(distances)
    min_distance = min((x for x in distances if x != 0))
        # Solves partial assignment and then performs rounding to get integral assignment
    epsilon = 0.1
    g_opt = min_distance + 0.0000001

    output = defaultdict( dict )
    t1 = time.monotonic()
    i = 0
    while g_opt <= max_distance:

        res, need_to_reassign = fair_partial_assignment(df, cluster_centers, fp_color_flag, fp_attributes, t, g_opt)

        if res["success"] == 2:

            output[i] = {}

            output[i]["Opt_cutoff"] = g_opt
            output[i]["Nodes to Reassign"] = need_to_reassign


            output[i]["num_clients"] = len(df)

            output[i]["num_clusters"] = num_clusters

            output[i]["num_colors"] = len(attributes[attr])

            output[i]["t_value"] = float(t)

            output[i]["success"] = res["success"]

            output[i]["dataset_distribution"] = dataset_ratio

            output[i]["unfair_score"] = initial_score

            output[i]["fair_score"] = res["fair_cost"]

            output[i]["partial_fair_score"] = res["partial_objective"]

            output[i]["sizes"] = sizes

            output[i]["attributes"] = attributes

            output[i]["ratios"] = ratios

            output[i]["centers"] = [list(center) for center in cluster_centers]

            output[i]["points"] = [list(point) for point in df.values]

            output[i]["assignment"] = res["assignment"]

            output[i]["partial_assignment"] = res["partial_assignment"]

            output[i]["name"] = dataset

            output[i]["cluster_time"] = cluster_time

            # output[i]["color_per_centre"] = res["color_per_centre"]

            output[i]["color_per_centre_vanilla"] = color_per_centre_vanilla

        g_opt = g_opt * (1 + epsilon)
        i+=1


    t2 = time.monotonic()
    fair_time = t2 - t1
    if len(output) > 0:
        min_obj = 10000000000
        min_obj_index = -1
        for index, res in output.items():
            if res["fair_score"] < min_obj:
                min_obj = res["fair_score"]
                min_obj_index = index

        save_file_string = dataset +  "_" + str(trial_number) + "_" +str(output[min_obj_index]["num_clients"])  + "_" + str(output[min_obj_index]["num_clusters"]) + "_" + str(output[min_obj_index]["num_colors"]) + "_" + str(output[min_obj_index]["t_value"]) + "_" + str(i)

        os.makedirs("./" + save_file_string + "/", exist_ok=True)

        out_file = open( "./" + save_file_string + "/output.json", "w" )
        output[min_obj_index]["fair_time"] = fair_time
        json.dump(output[min_obj_index], out_file)

        # df_plot = pd.DataFrame(color_per_centre_vanilla[attr])
        # df_plot.transpose().plot.bar()
        #
        # plt.savefig("./" + save_file_string + "/vanilla.png")
        # plt.close()
        # df_plot = pd.DataFrame(output[min_obj_index]["color_per_centre"])
        # df_plot.transpose().plot.bar()
        # plt.savefig("./" + save_file_string + "/fair.png")
        # plt.close()


