# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
import numpy as np
import sys
import gzip
import subprocess
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

def filter_downsample_fasta(input_fasta_path, output_fasta_path, min_length, downsample_n):
    # Decompress using zcat
    with open(input_fasta_path, "rb") as f_in, open(output_fasta_path, "w") as f_out:
        seqtk_path = Path("bin/seqtk/seqtk")
        # Unzip, filter and downsample
        zcat = subprocess.Popen(["zcat"], stdin=f_in, stdout=subprocess.PIPE)
        seqtk = subprocess.Popen([seqtk_path, "seq", f"-L {min_length}"], stdin=zcat.stdout, stdout=subprocess.PIPE)
        seqtk_sample = subprocess.Popen([seqtk_path, "sample", "-s100", "-", f"{downsample_n}"], stdin=seqtk.stdout, stdout=f_out)
        seqtk_sample.communicate()


def get_viral_contigs(number: int = 1000, min_length: int = 0):

    """ Takes a number of sequences and optional filter size.
        Returns a path to a resulting fasta file.
    """

    input = Path("data/raw/03_viral_sequences")
    output_root = Path("data/processed/05_viral_sequences")
    output_root.mkdir(parents=True, exist_ok=True)

    # Filter on lengths:
    
    min_length = 0
    downsample_count = number
    
    # When additional elements are added, this should be the basis for the output.
    final_sample_count = 200
    rerun = True

    if len(list(input.glob("*.fa.gz"))) < 1:
        logging.info("No samples found.")
        sys.exit(0)

    for sample in input.glob("*.fa.gz"):
        output_file = Path(output_root, sample.with_suffix('').with_suffix(".filter.fa").name)
        if output_file.exists() and output_file.stat().st_size > 0 and not rerun:
            logging.info(f"Skipping {sample.name}, already exists.")
            continue
        else:
            logging.info(f"Processing {sample.name}")
            filter_downsample_fasta(sample, output_file, min_length, downsample_count)
    

    # Combine all samples into one file:
    combined_fasta = Path(output_root, "combined.fa")

    if combined_fasta.exists() and combined_fasta.stat().st_size > 0 and not rerun:
        logging.info(f"Skipping {combined_fasta.name}, already exists.")
    else:
        logging.info(f"Combining all samples into {combined_fasta.name}")
        with open(combined_fasta, "w") as f_out:
            for sample in output_root.glob("*.filter.fa"):
                with open(sample, "r") as f_in:
                    f_out.write(f_in.read())
    

    return combined_fasta

    ### Maybe do something later

    # # Run mmseqs2 on all samples:

    # mmseqs_tmp = Path("./.tmp/mmseqs")
    # mmseqs_tmp.mkdir(parents=True, exist_ok=True)
    # mmseqs_clusters_path = Path(output_root, "mmseq_clusters")
    # mmseqs_clusters_path.mkdir(parents=True, exist_ok=True)
    # mmseq_clusters = Path(mmseqs_clusters_path, "combined")

    # # Run mmseqs easy-linclust on combined samples:

    # cmd = ['mmseqs','easy-linclust', combined_fasta, mmseq_clusters.absolute().as_posix(), mmseqs_tmp.absolute().as_posix()]
    # subprocess_return = subprocess.run(cmd)
    # if subprocess_return.returncode != 0:
    #     raise Exception(f"Error running mmseqs2: {subprocess_return}")
    
    
    # if mmseqsDB.exists():
    #     logging.info(f"Skipping mmseqs2 createdb, {mmseqsDB / 'mmseqsDB'} already exists.")
    # else:
    #     logging.info(f"Running mmseqs2 createdb on {combined_fasta.name}")
    #     cmd = ["mmseqs", "createdb", combined_fasta, mmseqsDB]
    #     subprocess_return = subprocess.run(cmd)
    #     if subprocess_return.returncode != 0:
    #         raise Exception(f"Error running mmseqs2 createdb: {subprocess_return}")
    
    # if mmseq_clusters.exists():
    #     logging.info(f"Skipping mmseqs2 cluster, {mmseq_clusters.name} already exists.")
    # else:
    #     logging.info(f"Running mmseqs2 cluster on {mmseq_clusters.name}")
    #     cmd = ['mmseqs','cluster', '--cluster-mode', f"{cluster_mode}", '--min-seq-id', f"{min_seq_id}", '--cov-mode', f"{cov_mode}",
    #         '-c', f"{min_cov}", mmseqsDB, mmseq_clusters.absolute().as_posix(), mmseqs_tmp.absolute().as_posix()]
        
    #     subprocess_return = subprocess.run(cmd)
    #     if subprocess_return.returncode != 0:
    #         raise Exception(f"Error running mmseqs2: {subprocess_return}")
    
    # if Path(str(mmseq_clusters)+'.tsv').exists():
    #     logging.info(f"Skipping mmseqs2 createtsv, {mmseq_clusters.absolute().as_posix()} already exists.")
    # else:
    #     logging.info(f"Running mmseqs2 createtsv on {mmseq_clusters.absolute().as_posix()}")
    #     cmd = ['mmseqs','createtsv', mmseqsDB, mmseqsDB, mmseq_clusters, Path(str(mmseq_clusters)+'.tsv')]
    #     subprocess_return = subprocess.run(cmd)

    #     if subprocess_return.returncode != 0:
    #         raise Exception(f"Error running mmseqs2: {subprocess_return}")
    

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    get_viral_contigs()