{
  "Opt_cutoff": The guessed OPT value used in this run,
  "num_clients": How many datapoints in the dataset,
  "num_clusters": How many clusters? k,
  "num_colors": How many colors? l,
  "t_value": What is the t-value here (fairness value),
  "success": 2 (was the LP successful? 2 if YES,
  "dataset_distribution": Dictionary with first level key being the fairness variable and second level being the colors, and how many datapoints we have in each color class,
  "unfair_score": Objective value of the Vanilla Clustering Phase,
  "fair_score": Objective Value after a fair assignment is found,
  "partial_fair_score": LP value of the first LP solved,
  "sizes": List of number of clients as indexed by color class,
  "attributes": Dictionary with first level key being the fairness variable and second level being colors and value being a list of indices of the datapoints in that color
  "ratios": Dictionary with first level key being the fairness variable and second level being centre indices and value is a list of ratios of color classes assigned to that centre 
  "centers": Coordinates of the Centres,
  "points": List of the datapoints,
  "assignment": A list of size num_clients * num_clusters, every consecutive num_clusters entries indicates which center the datapoint is assigned (so sums to 1, and it is integral so one of them is exactly 1 and rest are 0)
  "partial_assignment": Same as above but is a fractional assignment as the solution to the first LP,
  "name": name of the dataset used,
  "fair_time": How long it took to JUST making the vanilla solution into a fair one (2 LP + unassign and reassign + another LP),
  "cluster_time": How long it took to do the vanilla clustering,
  "color_per_centre": 2D dictionary, first keys are centre indices, second is color class, and value is how many clients of what color class is in which centre ,
  "color_per_centre_vanilla": Same as above but just after vanilla clustering
}