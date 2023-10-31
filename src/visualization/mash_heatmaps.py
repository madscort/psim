from pathlib import Path
import subprocess
import pandas as pd
import numpy as np
import sys
from tempfile import TemporaryDirectory

np.random.seed(1)

def sketch(input_fasta: [Path], output_sketch: Path):
    # Take list of fasta files, create mash sketch
    cmd = ['mash', 'sketch', '-o', output_sketch] + input_fasta
    subprocess.run(cmd, stdout=subprocess.PIPE)
    return output_sketch.with_suffix(".msh")

def mash(input_fasta: Path, sketchDB: Path):
    # Take path to fasta file and path to sketch database,
    # calculate mash distance to sketchDB.
    cmd = ['mash', 'dist', sketchDB, input_fasta]
    result = subprocess.run(cmd, stdout=subprocess.PIPE)
    return result.stdout.decode('utf-8')

# Sketch all sequences into a single database
# Run each sample against the database

# First get X random sequences from sampletable:
x = 50
sampletable = Path("data/processed/10_datasets/phage_25_fixed_25000_reduced_90_ws/sampletable.tsv")

df_sampletable = pd.read_csv(sampletable, sep="\t", header=None, names=['id', 'type', 'label'])
df_ps = df_sampletable.loc[df_sampletable['label'] == 1]['id'].values.tolist()
df_viral = df_sampletable.loc[(df_sampletable['label'] == 0) & (df_sampletable['type'] == 'viral_contigs')]['id'].values.tolist()
df_meta = df_sampletable.loc[(df_sampletable['label'] == 0) & (df_sampletable['type'] == 'meta_contigs')]['id'].values.tolist()

ps_seqs = [Path("data/processed/10_datasets/phage_25_fixed_25000_reduced_90_ws/sequences") / f"{seq}.fna" for seq in np.random.choice(df_ps, x, replace=False)]
viral_seqs = [Path("data/processed/10_datasets/phage_25_fixed_25000_reduced_90_ws/sequences") / f"{seq}.fna" for seq in np.random.choice(df_viral, x, replace=False)]
meta_seqs = [Path("data/processed/10_datasets/phage_25_fixed_25000_reduced_90_ws/sequences") / f"{seq}.fna" for seq in np.random.choice(df_meta, x, replace=False)]
combined_seqs = ps_seqs + viral_seqs + meta_seqs

contig_types = {'phage_satellite': ps_seqs,
         'viral_contigs': viral_seqs,
         'meta_contigs': meta_seqs}

# Sketch ps_seqs and save to tmp dir:

mashout = Path("data/visualization/mash_map/mash_dist_original.tsv")
mashout.parent.mkdir(parents=True, exist_ok=True)

with TemporaryDirectory() as tmp:
    sketch_file = sketch(combined_seqs, Path(tmp) / "sketchDB")
    # Run each sample against the database
    compared = set()
    with open(mashout, "w") as f:
        for type in contig_types:
            print(f"Running mash for {type}")
            for seq in contig_types[type]:
                mash_result = mash(seq, sketch_file).split("\n")
                for line in mash_result:
                    if line == "":
                        continue
                    item = line.strip().split("\t")
                    var1 = Path(item[0]).stem
                    var2 = Path(item[1]).stem
                    dist = item[2]
                    if var1 in compared:
                        continue
                    print(f"{var1}\t{var2}\t{dist}\t{type}", file=f)
                compared.add(seq)
        