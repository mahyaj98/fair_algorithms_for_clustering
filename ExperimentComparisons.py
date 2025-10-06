import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

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

dataset = "census"

# Average over

dir_list = [exp if exp.startswith(dataset) else None for exp in os.listdir("./")]

experiment_results = combine_results(dir_list)



df_cost = pd.DataFrame(experiment_results, columns=["unfair_score", "fair_score", "num_clients", "t_value", "num_colors", "num_clusters"])


client_count = df_cost['num_clients'].unique()

df_filtered = df_cost[df_cost['num_clients'] >= 2000 ]
df_filtered = df_filtered[df_filtered['num_clients'] <= 5000 ]

stats_df = df_filtered.groupby(['num_clients', 'num_clusters']).agg({
    'unfair_score': ['mean', 'std', lambda x: stats.t.interval(0.95, len(x)-1, loc=np.mean(x), scale=stats.sem(x))[0],
                     lambda x: stats.t.interval(0.95, len(x)-1, loc=np.mean(x), scale=stats.sem(x))[1]],
    'fair_score': ['mean', 'std', lambda x: stats.t.interval(0.95, len(x)-1, loc=np.mean(x), scale=stats.sem(x))[0],
                   lambda x: stats.t.interval(0.95, len(x)-1, loc=np.mean(x), scale=stats.sem(x))[1]]
}).reset_index()
stats_df.columns = ['num_clients', 'num_clusters',
                   'unfair_mean', 'unfair_std', 'unfair_ci_low', 'unfair_ci_high',
                   'fair_mean', 'fair_std', 'fair_ci_low', 'fair_ci_high']

# Create visualization
# df_time = pd.DataFrame(experiment_results, columns=["cluster_time", "fair_time", "num_clients", "t_value", "num_colors", "num_clusters"])
# df_time["total_fair_time"] = df_time["fair_time"] + df_time["cluster_time"]
# df_filtered_time = df_time[df_time['num_clients'] >= 2000 ]
# df_filtered_time = df_filtered_time[df_filtered_time['num_clients'] <= 5000 ]

plt.figure(figsize=(12, 6))

# Plot for unfair scores
# plt.subplot(1, 2, 1)
# for client_size in stats_df['num_clusters'].unique():
#     client_data = stats_df[stats_df['num_clusters'] == client_size]
#     plt.plot(client_data['num_clients'], client_data['unfair_mean'],
#              marker='o', label=f'Clusters: {client_size}')
#     plt.fill_between(client_data['num_clients'],
#                     client_data['unfair_ci_low'],
#                     client_data['unfair_ci_high'],
#                     alpha=0.2)
# plt.title('Unfair Score by Client Count')
# plt.xlabel('Number of Clients')
# plt.ylabel('Cost')
# plt.legend()
#
# # Plot for fair scores
# plt.subplot(1, 2, 2)
# for client_size in stats_df['num_clusters'].unique():
#     client_data = stats_df[stats_df['num_clusters'] == client_size]
#     plt.plot(client_data['num_clients'], client_data['fair_mean'],
#              marker='o', label=f'Clusters: {client_size}')
#     plt.fill_between(client_data['num_clients'],
#                     client_data['fair_ci_low'],
#                     client_data['fair_ci_high'],
#                     alpha=0.2)
# plt.title('Fair Score by Client Count')
# plt.xlabel('Number of Clients')
# plt.ylabel('Cost')
# plt.legend()
#
# plt.tight_layout()
# plt.savefig(f'{dataset}_cost_analysis.png', dpi=300, bbox_inches='tight')
# plt.show()
# plt.close()

plt.figure(figsize=(12, 6))

# Plot for unfair scores with error bars
plt.subplot(1, 2, 1)
for client_size in stats_df['num_clusters'].unique():
    client_data = stats_df[stats_df['num_clusters'] == client_size]
    plt.errorbar(client_data['num_clients'],
                client_data['unfair_mean'],
                yerr=client_data['unfair_std'],
                fmt='o-',  # format string ('o' for dots, '-' for lines)
                capsize=5,  # adds caps to the error bars
                label=f'Number of Clusters: {client_size}')
plt.title('Cost of vanilla clustering with Error Bars')
plt.xlabel('Number of Clients')
plt.ylabel('Cost')
plt.legend()

# Plot for fair scores with error bars
plt.subplot(1, 2, 2)
for client_size in stats_df['num_clusters'].unique():
    client_data = stats_df[stats_df['num_clusters'] == client_size]
    plt.errorbar(client_data['num_clients'],
                client_data['fair_mean'],
                yerr=client_data['fair_std'],
                fmt='o-',
                capsize=5,
                label=f'Number of Clusters: {client_size}')
plt.title('Cost of our fair algorithm with Error Bars')
plt.xlabel('Number of Clients')
plt.ylabel('Cost')
plt.legend()

plt.tight_layout()
plt.savefig(f'{dataset}_cost_with_errorbars.png', dpi=300, bbox_inches='tight')
plt.show()


# meantime_df = df_time.groupby(['num_clients', 'num_clusters'])[["cluster_time", 'total']].mean().reset_index()
# mean_df.to_excel(f"{dataset}_means.xlsx", index=False)
# df_cost.to_excel(f"{dataset}_costs.xlsx", index=False)

for k in [5,10,15,20]:

    df_cost_tmp = pd.DataFrame()
    # for n in client_count:
    #     tmp = df_cost.loc[(df_cost['num_clusters']==k) & (df_cost["num_clients"]==n)].sort_values( 'fair_score', ascending=True ).drop_duplicates(subset=['num_clients'], keep='first')
    #     df_cost_tmp = pd.concat([df_cost_tmp, tmp])
    #
    # df_cost_tmp = df_cost_tmp.sort_values('num_clients', ascending=True)
    df_cost_tmp = stats_df.loc[stats_df['num_clusters']==k].sort_values('num_clients', ascending=True)
    df_cost_tmp.plot( x="num_clients", y=["unfair_mean", "fair_mean"],
                      kind="line", style=["r--", "b--"], xlabel="Number of clients", ylabel=["Cost"],
                      label=["Vanilla", "Fair"] )
    plt.suptitle( "census1990_ss", fontsize=14, fontweight='bold' )

    plt.savefig( dataset +"_k=" + str(k) +"_cost_comparison_avg.png" )
    plt.close()

    # df_time_tmp = pd.DataFrame()
    # # for n in client_count:
    # #     tmp = df_time.loc [(df_time ['num_clusters'] == k) & (df_time ["num_clients"] == n)].sort_values( 'total_fair_time',
    # #                                                                                                       ascending=True ).drop_duplicates(
    # #         subset=['num_clients'], keep='first' )
    # #     df_time_tmp = pd.concat( [df_time_tmp, tmp] )
    # #
    # # df_time_tmp = df_time_tmp.sort_values( 'num_clients', ascending=True )
    # df_cost_tmp = meantime_df.loc[meantime_df['num_clusters']==k]
    # df_time_tmp.plot( x="num_clients", y=["cluster_time", "total_fair_time"],
    #               kind="line", style=["r--", "b--"], xlabel="Number of clients", ylabel=["Time(Seconds)"],
    #               label=["Vanilla Clustering", "Vanilla Clustering + Fair Algorithm"] )
    # plt.suptitle( "adult_race", fontsize=14, fontweight='bold' )
    # plt.savefig( dataset + "_k=" + str( k ) + "_time_comparison.png" )
    # plt.close()



if not stats_df.empty:
    fixed_num_clients = stats_df['num_clients'].iloc[0]
    fixed_num_clusters = stats_df['num_clusters'].iloc[0]

    print(f"Analyzing for fixed num_clients={fixed_num_clients} and num_clusters={fixed_num_clusters}")

    df_t_filtered = df_cost[(df_cost['num_clients'] == fixed_num_clients) & (df_cost['num_clusters'] == fixed_num_clusters)]

    if not df_t_filtered.empty:
        stats_t_df = df_t_filtered.groupby(['t_value']).agg({
            'unfair_score': ['mean', 'std'],
            'fair_score': ['mean', 'std']
        }).reset_index()
        stats_t_df.columns = ['t_value', 'unfair_mean', 'unfair_std', 'fair_mean', 'fair_std']

        plt.figure(figsize=(10, 6))
        plt.errorbar(stats_t_df['t_value'],
                     stats_t_df['fair_mean'],
                     yerr=stats_t_df['fair_std'],
                     fmt='o-',
                     capsize=5,
                     label='Fair Score')
        plt.errorbar(stats_t_df['t_value'],
                     stats_t_df['unfair_mean'],
                     yerr=stats_t_df['unfair_std'],
                     fmt='o-',
                     capsize=5,
                     label='Unfair Score')

        plt.title(f'Cost vs. t-value for {fixed_num_clients} Clients and {fixed_num_clusters} Clusters')
        plt.xlabel('t-value')
        plt.ylabel('Cost')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{dataset}_cost_vs_t_k={fixed_num_clusters}_n={fixed_num_clients}.png', dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print("No data found for the selected configuration to compare t-values.")
else:
    print("No data in stats_df to choose configuration from.")


