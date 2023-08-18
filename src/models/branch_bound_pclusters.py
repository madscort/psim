import pulp
import pandas as pd

# Read the data
data = pd.read_csv("data/external/cluster4080DB.tsv", sep="\t", header=None, names=["Cluster", "Protein"])
data["Genome"] = data["Protein"].str.split("|").str[0]

# Create a dictionary to store how many unique genomes each cluster covers
cluster_coverage = data.groupby('Cluster')['Genome'].unique().to_dict()

# Total genomes
total_genomes = data["Genome"].nunique()

# Set up the optimization problem
prob = pulp.LpProblem("MinimumClusters", pulp.LpMinimize)

# Define a binary variable for each cluster that is 1 if the cluster is selected and 0 otherwise
cluster_vars = pulp.LpVariable.dicts("Cluster", cluster_coverage.keys(), 0, 1, pulp.LpBinary)

# Objective function: minimize the number of selected clusters
prob += pulp.lpSum(cluster_vars)

# Constraints
for genome in data["Genome"].unique():
    prob += pulp.lpSum(cluster_vars[cluster] for cluster, genomes in cluster_coverage.items() if genome in genomes) >= 1

# Solve the problem
prob.solve()

# Extract selected clusters
selected_clusters = [cluster for cluster, var in cluster_vars.items() if var.varValue == 1]

print("Selected clusters:", selected_clusters)
print("Number of clusters:", len(selected_clusters))
