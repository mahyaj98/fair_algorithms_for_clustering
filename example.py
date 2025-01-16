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
dataset = config[config_str].get("dataset")
clustering_config_file = config[config_str].get("config_file")
num_clusters = list(map(int, config[config_str].getlist("num_clusters")))
max_points = config[config_str].getint("max_points")


for n_clusters in num_clusters:
    fair_clustering(dataset, clustering_config_file, data_dir, n_clusters, max_points)
