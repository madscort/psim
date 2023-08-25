# -*- coding: utf-8 -*-
import click
import logging
import subprocess
import sys
import gzip
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src.models.set_cover import sc_greedy
from Bio import SeqIO
from src.data.vir_contig_sampling import get_viral_contigs
from src.data.meta_contig_sampling import get_meta_contigs


def create_neg_sampletable(size: int = 10000, distribution: tuple = (0.9, 0.1)):
    
    total_samplesize = size

    # Distribution of sample types:
    distribution = distribution

    meta_contigs = get_meta_contigs(number = np.ceil(distribution[0]*total_samplesize), min_length=2500)
    vir_contigs = get_viral_contigs(number = np.ceil(distribution[1]*total_samplesize))
    types = {"meta_contigs": meta_contigs,
             "vir_contigs": vir_contigs}

    # meta_contigs = Path("data/processed/04_metagenomic_contigs/background/combined.fa")
    # vir_contigs = Path("data/processed/03_viral_sequences/combined.fa")

    output_sampletable = Path("data/processed/neg_sampletable.tsv")

    with open(output_sampletable, "w") as out_f:
        for seq_type in types.keys():
            with open(types[seq_type], "r") as f:
                records = list(SeqIO.parse(f, "fasta"))
                for record in records:
                    out_f.write(f"{record.id}\t{seq_type}\t0\n")

    return output_sampletable, types

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    create_neg_sampletable()