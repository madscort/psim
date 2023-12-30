
from pathlib import Path
import sys
import pandas as pd
# 2023-11-22 mads
# Simple script to reduce information on sampletypes.

# Input
dataset_root = Path("data/processed/10_datasets/")
version = "dataset_v02"
dataset = dataset_root / version
before_reduc = Path("data/processed/01_combined_databases/sample_table.tsv")
outf = Path("data/visualization/sample_type_dist_v02")
outfn = outf / "sample_raw_dist.tsv"
outfn_t = outf / "sample_type_dist.tsv"
outfn_g = outf / "sample_genus_dist.tsv"
outfn_ft = outf / "sample_family_type_dist.tsv"
outfn_gt = outf / "sample_genus_type_dist.tsv"
outfn_st = outf / "sample_species_type_dist.tsv"
outf.mkdir(parents=True, exist_ok=True)

tax = Path("data/processed/03_taxonomical_annotation/ps_tax_info.tsv")
tax_df = pd.read_csv(tax, sep="\t", header=0, names=["id", "family", "genus", "species"])

tables = ["train.tsv", "val.tsv", "test.tsv"]

# # Satellitetypes before after homology reduction:

# with open(outfn, "w") as fout:
#     for table in tables:
#         with open(dataset / table, "r") as fin:
#             fin.readline()
#             for line in fin:
#                 id, type, label = line.strip().split("\t")
#                 if label == "1":
#                     print(f"{id}\t{type}\t{label}\tafter", file=fout)
#         with open(before_reduc, "r") as fin:
#             for line in fin:
#                 id, type, label = line.strip().split("\t")
#                 if label == "1":
#                     print(f"{id}\t{type}\t{label}\tbefore", file=fout)
        

# # Basic sample types dist:

# with open(outfn_t, "w") as fout:
#     for table in tables:
#         with open(dataset / table, "r") as fin:
#             fin.readline()
#             for line in fin:
#                 id, type, label = line.strip().split("\t")
#                 re_type = type
#                 if type.startswith("host") or type.startswith("pro"):
#                     re_type = type.split("_")[0]
#                 else:
#                     if not type.startswith("meta"):
#                         re_type = "satellite"
#                 print(f"{id}\t{re_type}\t{label}", file=fout)

# # GENUS distribution

# with open(outfn_g, "w") as fout:
#     for table in tables:
#         with open(dataset / table, "r") as fin:
#             fin.readline()
#             for line in fin:
#                 id, type, label = line.strip().split("\t")
#                 if type.startswith("host") or type.startswith("pro"):
#                     re_type = type.split("_")[0]
#                     genus = type.split("_")[1]
#                 elif not type.startswith("meta"):
#                     re_type = "satellite"
#                     try:
#                         genus = tax_df[tax_df["id"] == id]["genus"].values[0]
#                     except IndexError:
#                         print(f"IndexError: {id}")
#                         continue
#                 else:
#                     continue
#                 print(f"{id}\t{re_type}\t{label}\t{str(genus).capitalize()}", file=fout)

# Family vs. satellite_type distribution

with open(outfn_ft, "w") as fout:
    for table in tables:
        with open(dataset / table, "r") as fin:
            fin.readline()
            for line in fin:
                id, type, label = line.strip().split("\t")
                if label == "0":
                    continue
                try:
                    family = tax_df[tax_df["id"] == id]["family"].values[0]
                except IndexError:
                    print(f"IndexError: {id}")
                    continue
                print(f"{id}\t{type}\t{label}\t{str(family)}", file=fout)

# GENUS vs. satellite_type distribution

# with open(outfn_gt, "w") as fout:
#     for table in tables:
#         with open(dataset / table, "r") as fin:
#             fin.readline()
#             for line in fin:
#                 id, type, label = line.strip().split("\t")
#                 if label == "0":
#                     continue
#                 try:
#                     genus = tax_df[tax_df["id"] == id]["genus"].values[0]
#                 except IndexError:
#                     print(f"IndexError: {id}")
#                     continue
#                 print(f"{id}\t{type}\t{label}\t{str(genus).capitalize()}", file=fout)

# # Species vs. satellite_type distribution

# with open(outfn_st, "w") as fout:
#     for table in tables:
#         with open(dataset / table, "r") as fin:
#             fin.readline()
#             for line in fin:
#                 id, type, label = line.strip().split("\t")
#                 if label == "0":
#                     continue
#                 try:
#                     species = tax_df[tax_df["id"] == id]["species"].values[0]
#                 except IndexError:
#                     print(f"IndexError: {id}")
#                     continue
#                 print(f"{id}\t{type}\t{label}\t{str(species)}", file=fout)
