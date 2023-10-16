# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
import numpy as np
import sys
import gzip
import subprocess
from pathlib import Path
from io import StringIO
from dotenv import find_dotenv, load_dotenv
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

def filter_downsample_fasta(input_fasta_path, output_fasta_path, min_length, downsample_n):
    # Decompress using zcat
    with open(input_fasta_path, "rb") as f_in, open(output_fasta_path, "w") as f_out:
        zcat = subprocess.Popen(["gunzip", "-c"], stdin=f_in, stdout=subprocess.PIPE)
        seqkit_seq = subprocess.Popen(["seqkit", "seq", f"--min-len={min_length}"], stdin=zcat.stdout, stdout=subprocess.PIPE)
        seqkit_rmdup = subprocess.Popen(["seqkit", "rmdup", "-s"], stdin=seqkit_seq.stdout, stdout=subprocess.PIPE)
        seqkit_sample = subprocess.Popen(["seqkit", "sample", "-s", "100", "-n", f"{int(downsample_n)}"], stdin=seqkit_rmdup.stdout, stdout=subprocess.PIPE, text=True)
        seqkit_rename = subprocess.Popen(["seqkit", "rename"], stdin=seqkit_sample.stdout, stdout=f_out, text=True)
        seqkit_rename.communicate()


def get_viral_contigs(number: int = 1000, min_length: int = 0, input_fasta_path: Path = Path("data/raw/03_viral_sequences/all_phages.fa.gz"), output_root: Path = Path("data/processed/05_viral_sequences")):

    """ Takes a number of sequences and optional filter length.
        Returns a path to a resulting fasta file.
    """

    input_fasta_path = input_fasta_path
    output_root = output_root
    output_root.mkdir(parents=True, exist_ok=True)

    # Filter on lengths:
    
    min_length = min_length
    downsample_count = number

    output_file = Path(output_root, f"min_{min_length}_viral_contigs.fna")
    if output_file.exists() and output_file.stat().st_size > 0:
        logging.info(f"Skipping {input_fasta_path.name}, already exists.")
    else:
        logging.info(f"Processing {output_file.name}")
        filter_downsample_fasta(input_fasta_path, output_file, min_length, downsample_count)


    return output_file


def filter_downsample_fasta_stdout(input_fasta_path, min_length, downsample_n):
    # Decompress using zcat
    with open(input_fasta_path, "rb") as f_in:
        # Unzip, filter, downsample and rezip

        zcat = subprocess.Popen(["gunzip", "-c"], stdin=f_in, stdout=subprocess.PIPE)
        seqkit_seq = subprocess.Popen(["seqkit", "seq", f"--min-len={min_length}"], stdin=zcat.stdout, stdout=subprocess.PIPE)
        seqkit_rmdup = subprocess.Popen(["seqkit", "rmdup", "-s"], stdin=seqkit_seq.stdout, stdout=subprocess.PIPE)
        seqkit_sample = subprocess.Popen(["seqkit", "sample", "-s", "100", "-n", f"{int(downsample_n)}"], stdin=seqkit_rmdup.stdout, stdout=subprocess.PIPE, text=True)
        seqkit_rename = subprocess.Popen(["seqkit", "rename"], stdin=seqkit_sample.stdout, stdout=subprocess.PIPE, text=True)
        stdout, _ = seqkit_rename.communicate()
        
        return stdout

def save_sequences_in_chunks(sequences, length, output_path):
    with open(output_path, "w") as output_file:
        for seq in sequences:
            start = np.random.randint(0, len(seq) - length)
            end = start + length

            # Get random sequence of length "length"
            chunk = seq.seq[start:end]
            chunk_id = f"{seq.id}_s{length}_{start}-{end}"
            SeqIO.write([SeqIO.SeqRecord(chunk, id=chunk_id, description="")], output_file, "fasta")

def fixed_length_viral_sampling(number: int = 10000, length: int = 25000, input_fasta_path: Path = Path("data/raw/03_viral_sequences/all_phages.fa.gz"), output_root: Path = Path("data/processed/05_viral_sequences")):
    """ sample contigs at a fixed length
    """

    number = number
    length = length

    # Paths
    viral_seqs = input_fasta_path
    output_root = output_root
    output_root.mkdir(parents=True, exist_ok=True)
    output = Path(output_root, f"{length}b_viral_contigs.fna")
    output.parent.mkdir(parents=True, exist_ok=True)

    stdout = filter_downsample_fasta_stdout(viral_seqs, length, number)
    sequences = list(SeqIO.parse(StringIO(stdout), "fasta"))
    save_sequences_in_chunks(sequences, length, output)

    return output

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    get_viral_contigs()