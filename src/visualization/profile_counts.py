from pathlib import Path
import torch
import pandas as pd

# five counts: unknown, metagenome, provirus, host, satellite
# based on type.
# So output: id, type, unknown, metagenome, provirus, host, satellite

dataset_root = Path("data/processed/10_datasets/")
version = "dataset_v02"
dataset = dataset_root / version
data_splits = torch.load(dataset / "strings" / "allDB" / "dataset.pt", map_location="cpu")
outfn = Path("data/visualization/profile_function/profile_counts.tsv")
outfn.parent.mkdir(parents=True, exist_ok=True)
splits = ['train', 'val', 'test']
with open(outfn, "w") as fout:
    for split in splits:
        sequences = data_splits[split]['sequences']
        for n, sequence in enumerate(sequences):
            for protein in sequence:
                if protein.startswith("IMGVR"):
                    origin = "prophage"
                elif protein.startswith("PS"):
                    origin = "satellite"
                elif protein.startswith("N") or protein.startswith("D") or protein.startswith("C"):
                    origin = "host"
                elif protein.startswith("S"):
                    origin = "metagenome"
                else:
                    origin = "unknown"
                print(split,
                      origin,
                      "|".join(protein.rsplit("_", 1)),
                      sep="\t",
                      file=fout)