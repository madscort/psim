import click
import pandas as pd
from pathlib import Path


def sc_greedy(mmseq_clusters: Path = Path("data/processed/protein_clusters/cluster4080DB.tsv")):
    """
    Find protein clusters with highest coverage using a greedy set cover algorithm.
    """

    data = pd.read_csv(mmseq_clusters, sep="\t", header=None, names=["cluster", "protein"])

    # Extract the genome ID from the protein IDs (format: genome|protein)
    data["genome"] = data["protein"].str.split("|").str[0]

    # Compute the unique genomes each cluster covers
    cluster_to_genomes = data.groupby("cluster")["genome"].unique().to_dict()
    total_genomes = set(data["genome"].unique())

    # Initialize sets to keep track of covered genomes and selected clusters
    covered_genomes = set()
    selected_clusters = []

    while covered_genomes != total_genomes and cluster_to_genomes:
        # Calculate how many new genomes each cluster would add
        new_genomes_counts = {cluster: len(set(genomes) - covered_genomes) for cluster, genomes in cluster_to_genomes.items()}
        
        # Select the cluster that contributes the most new genomes
        best_cluster = max(new_genomes_counts, key=new_genomes_counts.get)
        
        # Update covered genomes and selected clusters
        covered_genomes.update(cluster_to_genomes[best_cluster])
        selected_clusters.append(best_cluster)
        
        # Remove the selected cluster from the dictionary
        del cluster_to_genomes[best_cluster]

    return selected_clusters, covered_genomes




