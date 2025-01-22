import configparser
import sys
from fair_clustering import fair_clustering
from util.configutil import read_list

config_file = "config/example_config.ini"
config = configparser.ConfigParser(converters={'list': read_list})
config.read(config_file)
config_str = "bank" if len(sys.argv) == 1 else sys.argv[1]

print("Using config_str = {}".format(config_str))


data_dir = config[config_str].get("data_dir")
datasets = ["adult_race", "census1990_ss", "creditcard_education"]
clustering_config_file = config[config_str].get("config_file")
num_clusters = [15, 10, 5]
num_clients = [2000, 2500, 3000, 3500, 4000, 4500]
trials = 5

for dataset in datasets:
    for max_points in num_clients:
        for n_clusters in num_clusters:
            for i in range( trials ):
                fair_clustering(dataset, clustering_config_file, data_dir, n_clusters, max_points, i)
