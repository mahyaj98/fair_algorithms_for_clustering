import configparser
import sys
from fair_clustering import fair_clustering
from util.configutil import read_list

config_file = "config/example_config.ini"
config = configparser.ConfigParser(converters={'list': read_list})
config.read(config_file)
config_str = "bank" if len(sys.argv) == 1 else sys.argv[1]


data_dir = config[config_str].get("data_dir")
datasets = [ "bank"]
clustering_config_file = config[config_str].get("config_file")
num_clusters = [10]
num_clients = [1000, 1500, 2000, 2500, 3000, 3500, 4000]
trials = 10

for n_clusters in num_clusters:
    for max_points in num_clients:
        for dataset in datasets:
            for i in range(trials):
                print(dataset + ", k = " + str(n_clusters) + ", n = " + str(max_points) + ", attempt = " + str(i) + " ")
                fair_clustering(dataset, clustering_config_file, data_dir, n_clusters, max_points, i)
