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
dataset = Path("data/processed/10_datasets/dataset_v01")
sampletable = dataset / "test.tsv"

# Just take satellite test sequeneces
df_sampletable = pd.read_csv(sampletable, sep="\t", header=0, names=['id', 'type', 'label'])
df_ps = df_sampletable.loc[df_sampletable['label'] == 1]['id'].values.tolist()

ps_seqs = [dataset / "test" / "sequences" / f"{seq}.fna" for seq in df_ps]

# Sketch ps_seqs and save to tmp dir:

mashout = Path("data/visualization/mash_map/mash_dist_ps_only.tsv")
mashout.parent.mkdir(parents=True, exist_ok=True)

with TemporaryDirectory() as tmp:
    sketch_file = sketch(ps_seqs, Path(tmp) / "sketchDB")
    # Run each sample against the database
    compared = set()
    with open(mashout, "w") as f:
        for seq in ps_seqs:
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
                type = df_sampletable.loc[df_sampletable['id'] == var1]['type'].values[0]
                print(f"{var1}\t{var2}\t{dist}\t{type}", file=f)
            compared.add(seq)
        