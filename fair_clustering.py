import configparser
import time
from collections import defaultdict
from functools import partial
import numpy as np
from sympy import ceiling

from cplex_fair_assignment_lp_solver import fair_partial_assignment
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
def fair_clustering(dataset, config_file, data_dir, num_clusters, deltas, max_points):
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
    min_rep = 1
    max_rep = 0
    for var, bucket_dict in attributes.items():
        representation[var] = {k : (len(bucket_dict[k]) / len(df)) for k in bucket_dict.keys()}
        if var != 'default':
            min_rep = min(min(representation[var].values()),min_rep)
            max_rep = max(max(representation[var].values()),max_rep)
    t = ceiling(max_rep/min_rep)
    # Select only the desired columns
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


    # dataset_ratio : Ratios for colors in the dataset
    dataset_ratio = {}
    for attr, color_dict in attributes.items():
        dataset_ratio[attr] = {int(color) : len(points_in_color) / len(df) 
                            for color, points_in_color in color_dict.items()}

    # fairness_vars (list[str]) : Variables to perform fairness balancing on
    fairness_vars = config[dataset].getlist("fairness_variable")

    fp_color_flag = take_by_key(color_flag, fairness_vars)
    fp_attributes = take_by_key(attributes, fairness_vars)
        # Solves partial assignment and then performs rounding to get integral assignment
    epsilon = 0.01
    g_opt = initial_score
    for i in range(0,100):

        t1 = time.monotonic()
        res = fair_partial_assignment(df, cluster_centers, fp_color_flag, fp_attributes, t, g_opt)
        t2 = time.monotonic()
        fair_time = t2 - t1
        g_opt = g_opt * (1 + epsilon)


        if res.get("success") == 'optimal':
            output = {}

            output["num_clusters"] = num_clusters

            output["success"] = res["success"]

            output["status"] = res["status"]

            output["dataset_distribution"] = dataset_ratio


            output["unfair_score"] = initial_score

            output["fair_score"] = res["objective"]

            output["partial_fair_score"] = res["partial_objective"]

            output["sizes"] = sizes

            output["attributes"] = attributes

            output["ratios"] = ratios


            output["centers"] = [list(center) for center in cluster_centers]
            output["points"] = [list(point) for point in df.values]
            output["assignment"] = res["assignment"]

            output["partial_assignment"] = res["partial_assignment"]

            output["name"] = dataset
            output["clustering_method"] = clustering_method
            output["scaling"] = scaling
            output["time"] = fair_time
            output["cluster_time"] = cluster_time


            time.sleep(1)
