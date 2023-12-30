from pathlib import Path
import sys

viral_fasta = Path("data/processed/05_viral_sequences/imgvr_filtered/cd.hit.IMGVR_minimal_seqs.90_DOWNSAMPLED.fna")
viral_info = Path("data/processed/05_viral_sequences/imgvr_filtered/IMGVR_all_filtered_seqs_info.tsv")

imgs = []
with open(viral_fasta, 'r') as fin:
    for line in fin:
        if line.startswith(">"):
            imgs.append(line.strip()[1:])

provirus = 0
linear = 0

with open(viral_info, 'r') as fin:
    for line in fin:
        id, species, type = line.strip().split("\t")
        if id in imgs:
            if type == "Provirus":
                provirus += 1
            elif type == "Linear":
                linear += 1
            else:
                print(f"Unknown type: {type}")

print(f"Number of IMG/VR sequences: {len(imgs)}")
print(f"Number of provirus: {provirus}")
print(f"Number of linear: {linear}")
print(f"Number of circular: {len(imgs) - provirus - linear}")