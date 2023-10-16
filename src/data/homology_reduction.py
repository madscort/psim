# -*- coding: utf-8 -*-
import click
import logging
import subprocess
import sys
import gzip
import pandas as pd
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from Bio import SeqIO

def cluster_reduce(identity: float = 0.9, input_data: Path = None, output_path: Path = None):
    """ Use CD-HIT-est to reduce homology by clustering on sequence identity """

    input_sequences = Path(input_data, "all_sequences.fna")
    output_folder = Path(input_data, f"reduced_{int(identity*100)}")
    output_sequences = Path(output_folder, "all_sequences.fna")
    output_sequences.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"Reducing homology in {input_sequences} to {output_sequences} with identity {identity}")

    # Run CD-HIT-est
    # run_cd_hit(input_sequences, identity, output_sequences)

    # Get representative sequences identifiers

    identifiers = []
    with open(output_sequences, "r") as f:
        for record in SeqIO.parse(f, "fasta"):
            identifiers.append(record.id)
    if len(identifiers) == len(set(identifiers)):
        logging.info(f"Reduced sequences to {len(set(identifiers))} with identity {identity}")
    else:
        logging.info(f"Reduced sequences to {len(set(identifiers))} with identity {identity} BUT {len(identifiers) - len(set(identifiers))} duplicates were found!")
        return
    
    # Reduce sampletable based on output:
    sampletable = Path(input_data, "sample_table.tsv")
    output_sampletable = Path(output_folder, "sample_table.tsv")

    with open(sampletable, "r") as f, open(output_sampletable, "w") as out:
        for line in f:
            identifier = line.split("\t")[0]
            if identifier in identifiers:
                out.write(line)

    # Reduce coordinates based on output:
    coord_in = Path(input_data, "satellite_coordinates.tsv")
    coord_out = Path(output_folder, "satellite_coordinates.tsv")

    with open(coord_in, "r") as f, open(coord_out, "w") as out:
        for line in f:
            identifier = line.split("\t")[0]
            if identifier in identifiers:
                out.write(line)
    
    # Reduce protein sequences using SeqIO based on output:
    proteins_in = Path(input_data, "all_proteins.faa")
    proteins_out = Path(output_folder, "all_proteins.faa")

    with open(proteins_in, "r") as f, open(proteins_out, "w") as out:
        for record in SeqIO.parse(f, "fasta"):
            if record.id in identifiers:
                SeqIO.write(record, out, "fasta")
    
    # Symlink to reference sequences:
    ref_sequences = Path(input_data, "all_reference_sequences.fna")
    ref_sequences_out = Path(output_folder, "all_reference_sequences.fna")
    ref_sequences_out.symlink_to(ref_sequences)

    logging.info("Done!")

def run_cd_hit(input_fasta_path: Path = None, identity: float = None, output_prefix: Path = None):
    """ Run CD-HIT-est on a fasta file """
    cmd = ["cd-hit-est", "-i", input_fasta_path, "-o", output_prefix, "-c", str(identity), "-n", "9"]
    subprocess.run(cmd)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    cluster_reduce(0.80, Path("/Users/madsniels/Documents/_DTU/speciale/cpr/code/psim/data/processed/01_combined_renamed"))