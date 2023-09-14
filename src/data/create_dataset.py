# -*- coding: utf-8 -*-
import click
import logging
import subprocess
import sys
import gzip
import pandas as pd
import numpy as np
from src.data.sat_contig_sampling import sat_contig_sampling
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from Bio import SeqIO
from src.data.vir_contig_sampling import get_viral_contigs, fixed_length_viral_sampling
from src.data.meta_contig_sampling import get_meta_contigs, fixed_length_contig_sampling

def create_dataset():
    input = Path("/Users/madsniels/Documents/_DTU/speciale/cpr/code/psim/data/processed/01_combined_renamed/reduced_90")
    root = Path("data/processed/10_datasets")
    dataset_id = "phage_25_fixed_25000_reduced_90"
    dataset_root = Path(root, dataset_id)
    tmp = Path(dataset_root, ".tmp")
    tmp.mkdir(parents=True, exist_ok=True)

    fixed_length = True
    length = 25000
    negative_samplesize = 2000

    # Distribution of negative sample types (metagenomic, phage sequence):
    distribution = (0.75, 0.25)

    # Get satellites:
    sat_contig_sampling(fixed=fixed_length,
                        fixed_length=length,
                        sanity_check=True,
                        input_root=input,
                        output_root=tmp)
    
    ## Get negative samples

    # Get metagenomic contigs:
    if fixed_length:
        meta_contigs = fixed_length_contig_sampling(number = np.ceil(distribution[0]*negative_samplesize),
                                                    length=length,
                                                    output_root=tmp)
    else:
        meta_contigs = get_meta_contigs(number = np.ceil(distribution[0]*negative_samplesize),
                                        min_length=2500,
                                        output_root=tmp)

    # Get viral contigs:
    if fixed_length:
        vir_contigs = fixed_length_viral_sampling(number = np.ceil(distribution[1]*negative_samplesize),
                                                  length=length,
                                                  output_root=tmp)
    else:
        vir_contigs = get_viral_contigs(number = np.ceil(distribution[1]*negative_samplesize),
                                        output_root=tmp)

    # Create sampletable:
    sampletable = Path(dataset_root, "sampletable.tsv")

    sequence_collection = {'meta_contigs': meta_contigs,
                           'viral_contigs': vir_contigs}

    with open(sampletable, "w") as out_f, open(Path(tmp, "sample_table.tsv"), "r") as f:
        for line in f:
            out_f.write(line)
        for seq_type in sequence_collection:
            with open(sequence_collection[seq_type], "r") as f:
                records = list(SeqIO.parse(f, "fasta"))
                for record in records:
                    out_f.write(f"{record.id}\t{seq_type}\t0\n")
    
    # Concatenate all fasta files in tmp:
    fasta = Path(dataset_root, "dataset.fna")
    with open(fasta, "w") as out_f:
        for f in tmp.glob("*.fna"):
            with open(f, "r") as in_f:
                for line in in_f:
                    out_f.write(line)

    # Create a folder with a file per sequence:
    sequences = Path(dataset_root, "sequences")
    sequences.mkdir(parents=True, exist_ok=True)
    for record in SeqIO.parse(fasta, "fasta"):
        with open(Path(sequences, f"{record.id}.fna"), "w") as out_f:
            SeqIO.write(record, out_f, "fasta")



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    create_dataset()