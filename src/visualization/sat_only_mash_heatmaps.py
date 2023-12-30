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
dataset = Path("data/processed/10_datasets/dataset_v02")
sampletable = dataset / "test.tsv"

# Just take satellite test sequeneces
df_sampletable = pd.read_csv(sampletable, sep="\t", header=0, names=['id', 'type', 'label'])
df_ps = df_sampletable['id'].values.tolist()

ps_seqs = [dataset / "test" / "sequences" / f"{seq}.fna" for seq in df_ps]
filtered_ps_seqs = []
host_count = 0
provirus_count = 0
satellite_count = 0
metagenome_count = 0

for seq in ps_seqs:
    if str(seq.stem).startswith("IMGVR"):
        if provirus_count > 100:
            continue
        else:
            filtered_ps_seqs.append(seq)
        provirus_count += 1
    elif str(seq.stem).startswith("PS"):
        if satellite_count > 100:
            continue
        else:
            filtered_ps_seqs.append(seq)
        satellite_count += 1
    elif str(seq.stem).startswith("N") or str(seq.stem).startswith("D") or str(seq.stem).startswith("C"):
        if host_count > 100:
            continue
        else:
            filtered_ps_seqs.append(seq)
        host_count += 1
    elif str(seq.stem).startswith("S"):
        if metagenome_count > 100:
            continue
        else:
            filtered_ps_seqs.append(seq)
        metagenome_count += 1

# Sketch ps_seqs and save to tmp dir:

mashout = Path("data/visualization/mash_map/mash_dist_all_AND_all.tsv")
mashout.parent.mkdir(parents=True, exist_ok=True)

with TemporaryDirectory() as tmp:
    sketch_file = sketch(filtered_ps_seqs, Path(tmp) / "sketchDB")
    # Run each sample against the database
    compared = set()
    with open(mashout, "w") as f:
        for seq in filtered_ps_seqs:
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
                #type = df_sampletable.loc[df_sampletable['id'] == var1]['type'].values[0]
                origin = "unknown"
                if var1.startswith("IMGVR"):
                    origin = "provirus"
                elif var1.startswith("PS"):
                    origin = "satellite"
                elif var1.startswith("N") or var1.startswith("D") or var1.startswith("C"):
                    origin = "host"
                elif var1.startswith("S"):
                    origin = "metagenome"
                print(f"{var1}\t{var2}\t{dist}\t{origin}", file=f)
            compared.add(seq)
        