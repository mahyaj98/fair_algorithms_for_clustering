import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def combine_results(dir_list):
    results = []
    try:
        for exp in dir_list:
            if exp is not None:
                file_name = "./" + exp + "/output.json"
                f = open( file_name , "r" )
                data = json.load( f )
                results.append( data )
    except FileNotFoundError:
        pass

    return results

dataset = "census1990_ss"

dir_list = [exp if exp.startswith(dataset) else None for exp in os.listdir("./")]

experiment_results = combine_results(dir_list)

df_cost = pd.DataFrame(experiment_results, columns=["unfair_score", "fair_score", "num_clients"])
df_cost = df_cost.sort_values('num_clients', ascending=True)

df_cost.plot(x="num_clients", y=["unfair_score", "fair_score"],
        kind="line", style=["r--","b--"], xlabel="Number of clients", ylabel=["Cost"], label = ["Vanilla","Fair"])

plt.savefig(dataset + "_cost_comparison.png")


df_time = pd.DataFrame(experiment_results, columns=["cluster_time", "fair_time", "num_clients"])
df_time["total_fair_time"] = df_time["fair_time"] + df_time["cluster_time"]
df_time = df_time.sort_values('num_clients', ascending=True)

df_time.plot(x="num_clients", y=["cluster_time", "total_fair_time"],
        kind="line", style=["r--","b--"], xlabel="Number of clients", ylabel=["Time"], label = ["Vanilla Clustering","Vanilla Clustering + Fair Algorithm"])

plt.savefig(dataset + "_time_comparison.png")

