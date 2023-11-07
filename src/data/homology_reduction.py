# -*- coding: utf-8 -*-
import click
import logging
import subprocess
import sys
import gzip
import pandas as pd
import collections
from pathlib import Path
from tempfile import TemporaryDirectory
from dotenv import find_dotenv, load_dotenv
from Bio import SeqIO


# mads 2023-10-23
# Script for homology reduction of samples in sampletable + various utilty functions.
# Takes a sampletable and corresponding fasta file as input
# Outputs a new sampletable with reduced homology

def cluster_reduce(identity: float = 0.9, input_data: Path = None, output_path: Path = None):
    """ Use CD-HIT-est to reduce homology by clustering on sequence identity """

    input_sequences = Path(input_data, "all_sequences.fna")
    output_folder = Path(input_data, f"reduced_{int(identity*100)}")
    output_sequences = Path(output_folder, "all_sequences.fna")
    output_sequences.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"Reducing homology in {input_sequences} to {output_sequences} with identity {identity}")

    # Run CD-HIT-est
    run_cd_hit(input_sequences, identity, output_sequences)

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
    
    input_sampletable = Path("data/processed/02_preprocessed_database/01_deduplication/sampletable.tsv")
    output_sampletable = Path("data/processed/02_preprocessed_database/02_homology_reduction/sampletable.tsv")
    input_fasta = Path("data/processed/01_combined_databases/all_sequences.fna")
    ps_sample = collections.namedtuple("ps_sample", ["sample_id", "type", "label"])
    identity = 0.90

    # Get samples
    samples = {}
    with open(input_sampletable, "r") as f:
        for line in f:
            line = line.strip().split("\t")
            samples[line[0]] = ps_sample(line[0], line[1], line[2])

    # Create temporary folder for CD-HIT-est

    with TemporaryDirectory() as tmp:
        # Create fasta file with all sequences:
        cd_hit_in = Path(tmp, "cd_hit_in.fna")
        cd_hit_out = Path(tmp, "cd_hit_out.fna")
        
        with open(input_fasta, "r") as f, open(cd_hit_in, "w") as out:
            for record in SeqIO.parse(f, "fasta"):
                if record.id in samples:
                    SeqIO.write(record, out, "fasta")

        # Run CD-HIT-est
        run_cd_hit(cd_hit_in, identity, cd_hit_out)

        # Parse output
        identifiers = []
        with open(cd_hit_out, "r") as f:
            for record in SeqIO.parse(f, "fasta"):
                identifiers.append(record.id)
        
    if len(identifiers) == len(set(identifiers)):
        logging.info(f"Reduced sequences to {len(set(identifiers))} with identity {identity}")
    else:
        logging.info(f"Reduced sequences to {len(set(identifiers))} with identity {identity} BUT {len(identifiers) - len(set(identifiers))} duplicates were found!")

    # Reduce sampletable based on output:
    with open(output_sampletable, "w") as out_f:
        for identifier in identifiers:
            out_f.write(f"{identifier}\t{samples[identifier].type}\t{samples[identifier].label}\n")

    #cluster_reduce(0.80, Path("/Users/madsniels/Documents/_DTU/speciale/cpr/code/psim/data/processed/01_combined_renamed"))