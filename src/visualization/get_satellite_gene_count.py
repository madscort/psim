from pathlib import Path
import sys
from collections import Counter
# First get direct info on gene count in each raw satellite sequence
# Easyliest obtainable from the pfama string model intermediate results.

proteins_file = Path("data/processed/01_combined_databases/all_proteins.faa")
sampletable = Path("data/processed/02_preprocessed_database/02_homology_reduction/sampletable.tsv")
output = Path("data/visualization/gene_counts/raw_satellite_gene_counts.tsv")

sample_type = {}
# Get sample type info:
with open(sampletable, 'r') as fin:
    for line in fin:
        id, type, _ = line.strip().split("\t")
        sample_type[id] = type

# Simple protein counter:
sample_id = []
with open(proteins_file, 'r') as fin:
    for line in fin:
        if line.startswith(">"):
            sample_id.append(line.strip().split("|")[0][1:])

protein_counts = Counter(sample_id)
with open(output, 'w') as fout:
    for k, v in protein_counts.items():
        if k not in sample_type:
            continue
        if k.startswith("PS_R"):
            ori = "Rocha"
        else:
            ori = "Trivedi"
        print(k, v, sample_type[k], ori, sep="\t", file=fout)

