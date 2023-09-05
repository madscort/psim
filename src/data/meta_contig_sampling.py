# -*- coding: utf-8 -*-
import click
import logging
import numpy as np
import sys
import subprocess
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from io import StringIO

np.random.seed(1)

def filter_downsample_fasta(input_fasta_path, output_fasta_path, min_length, downsample_n):
    # Decompress using zcat
    with open(input_fasta_path, "rb") as f_in, open(output_fasta_path, "w") as f_out:
        # Unzip, filter, downsample and rezip
        zcat = subprocess.Popen(["gunzip", "-c"], stdin=f_in, stdout=subprocess.PIPE)
        seqkit_seq = subprocess.Popen(["seqkit", "seq", f"--min-len={min_length}"], stdin=zcat.stdout, stdout=subprocess.PIPE)
        seqkit_rmdup = subprocess.Popen(["seqkit", "rmdup", "-s"], stdin=seqkit_seq.stdout, stdout=subprocess.PIPE)
        seqkit_sample = subprocess.Popen(["seqkit", "sample", "-s", "100", "-n", f"{int(downsample_n)}"], stdin=seqkit_rmdup.stdout, stdout=subprocess.PIPE, text=True)
        seqkit_rename = subprocess.Popen(["seqkit", "rename"], stdin=seqkit_sample.stdout, stdout=f_out, text=True)
        stdout, _ = seqkit_rename.communicate()
        
        return stdout

def get_meta_contigs(number: int = 1000, min_length: int = 2500,
                     output_root: Path = Path("data/processed/04_metagenomic_contigs/background"),
                     input_path: Path = Path("data/raw/02_metagenomic_samples/all_metagenomic_samples.fa.gz")):
    """ Takes a number of sequences and optional filter size.
        Returns a path to a resulting fasta file.
    """

    input_path = input_path
    output_root = output_root
    output_root.mkdir(parents=True, exist_ok=True)

    # Filter on lengths:
    
    min_length = min_length
    downsample_count = number
    
    output_file = Path(output_root, f"min_{min_length}_contigs.fna")
    if output_file.exists() and output_file.stat().st_size > 0:
        logging.info(f"Skipping {output_file.name}, already exists.")
    else:
        logging.info(f"Processing {output_file.name}")
        filter_downsample_fasta(input_path, output_file, min_length, downsample_count)
    
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

def save_sequences_in_chunks(sequences, size, output_path):
    with open(output_path, "w") as output_file:
        for seq in sequences:
            start = np.random.randint(0, len(seq) - size)
            end = start + size

            # Get random sequence of length "size"
            chunk = seq.seq[start:end]
            chunk_id = f"{seq.id}_s{size}_{start}-{end}"
            SeqIO.write([SeqIO.SeqRecord(chunk, id=chunk_id, description="")], output_file, "fasta")

def fixed_length_contig_sampling(number: int = 10000, length: int = 25000,
                                 output_root: Path = Path("data/processed/04_metagenomic_contigs/fixed_length"),
                                 input_path: Path = Path("data/raw/02_metagenomic_samples/all_metagenomic_samples.fa.gz")):
    """ sample contigs at a fixed length
    """

    number = number
    length = length

    # Paths
    contigs = input_path
    output_root = output_root
    output = Path(output_root, f"{length}b_contigs.fna")
    output.parent.mkdir(parents=True, exist_ok=True)

    stdout = filter_downsample_fasta_stdout(contigs, length, number)
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

    fixed_length_contig_sampling()