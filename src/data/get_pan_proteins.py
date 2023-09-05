# -*- coding: utf-8 -*-
import click
import logging
import subprocess
import sys
import pandas as pd
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src.models.set_cover import sc_greedy
from Bio import SeqIO


def get_records_from_identifiers(fasta: Path = Path("data/raw/proteins/all_proteins.fna"), identifiers: list = "GENOME_DBU_1|PROTEIN_1"):
    """
    Get record from a fasta file given one or a list of identifier.
    """
    
    records = list(SeqIO.parse(fasta, "fasta"))
    # create empty dictionary of identifiers to maintian order
    records_out = {identifier: None for identifier in identifiers}
    for record in records:
        if record.id in identifiers:
            records_out[record.id] = record
    
    records_out = [records_out[identifier] for identifier in identifiers]
    return records_out


def get_pan_proteins(protein_seqs: Path = Path("data/raw/proteins/all_proteins.fna"),
                     cluster_mode: int = 1,
                     cov_mode: int = 0,
                     min_cov: float = 0.9,
                     min_seq_id: float = 0.3,
                     rerun_all: bool = False,
                     out_prefix: str = "cluster3090DB",
                     out_file_root: Path = Path("data/processed/protein_clusters/test/cluster3090DB")):

    rerun_all = rerun_all
    
    cluster_mode = cluster_mode
    cov_mode = cov_mode
    min_cov = min_cov
    min_seq_id = min_seq_id

    mmseqs_tmp = Path("./.tmp/mmseqs")
    mmseqs_tmp.mkdir(parents=True, exist_ok=True)
    protein_seqs = protein_seqs

    out_prefix = out_prefix
    out_file_root = out_file_root

    proteinsDB = Path(out_file_root, "mmseqsDB", "proteinDB")
    mmseq_clusters = Path(out_file_root, "cluster", out_prefix)
    rep_protein_seqs = Path(out_file_root, "representatives", out_prefix).with_suffix(".fna")

    proteinsDB.parent.mkdir(parents=True, exist_ok=True)
    mmseq_clusters.parent.mkdir(parents=True, exist_ok=True)
    rep_protein_seqs.parent.mkdir(parents=True, exist_ok=True)

    """Create protein database for clustering with MMseqs2
    """

    if proteinsDB.exists() and not rerun_all:
        logging.info(f"Skipping mmseqs2 createdb, {proteinsDB.absolute().as_posix()} already exists.")
    else:
        proteinsDB.unlink(missing_ok=True)
        proteinsDB.parent.mkdir(parents=True, exist_ok=True)

        logging.info(f"Running mmseqs2 createdb on {proteinsDB.absolute().as_posix()}")

        cmd = ["mmseqs", "createdb", protein_seqs, proteinsDB]
        subprocess_return = subprocess.run(cmd, stdout=subprocess.DEVNULL)
        if subprocess_return.returncode != 0:
            raise Exception(f"Error running mmseqs2: {subprocess_return}")
    
    """Cluster proteins with MMseqs2
    """

    if Path(mmseq_clusters).with_suffix(".1").exists() and not rerun_all:
        logging.info(f"Skipping mmseqs2 cluster, {mmseq_clusters.absolute().as_posix()} already exists.")
    else:
        for ps in mmseq_clusters.parent.glob(str(mmseq_clusters.name) + "*"):
            ps.unlink(missing_ok=True)
        
        mmseq_clusters.parent.mkdir(parents=True, exist_ok=True)
        mmseqs_tmp.mkdir(parents=True, exist_ok=True)

        logging.info(f"Running mmseqs2 cluster on {mmseq_clusters.absolute().as_posix()}")

        cmd = ['mmseqs','cluster', '--cluster-mode', f"{cluster_mode}", '--min-seq-id', f"{min_seq_id}", '--cov-mode', f"{cov_mode}",
               '-c', f"{min_cov}", proteinsDB, mmseq_clusters, mmseqs_tmp]
        subprocess_return = subprocess.run(cmd)
        if subprocess_return.returncode != 0:
            raise Exception(f"Error running mmseqs2: {subprocess_return}")

    """Create tsv file with cluster and representative protein
    """

    if Path(str(mmseq_clusters)+'.tsv').exists() and not rerun_all:
        logging.info(f"Skipping mmseqs2 createtsv, {mmseq_clusters.absolute().as_posix()} already exists.")
    else:
        logging.info(f"Running mmseqs2 createtsv on {mmseq_clusters.absolute().as_posix()}")
        cmd = ['mmseqs','createtsv', proteinsDB, proteinsDB, mmseq_clusters, Path(str(mmseq_clusters)+'.tsv')]
        subprocess_return = subprocess.run(cmd, stdout=subprocess.DEVNULL)
        if subprocess_return.returncode != 0:
            raise Exception(f"Error running mmseqs2: {subprocess_return}")

    if rep_protein_seqs.exists() and not rerun_all:
        logging.info(f"Skipping set cover, {mmseq_clusters.absolute().as_posix()} already exists.")
    else:
        logging.info(f"Running set cover on {mmseq_clusters.absolute().as_posix()}")
        ordered_cluster_reps, _ = sc_greedy(mmseq_clusters = Path(str(mmseq_clusters)+'.tsv'))
        logging.info(f"Number of cluster representatives: {len(ordered_cluster_reps)}")
        records = get_records_from_identifiers(fasta=protein_seqs, identifiers=ordered_cluster_reps)
        SeqIO.write(records, rep_protein_seqs.absolute().as_posix(), "fasta")

    return rep_protein_seqs



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    get_pan_proteins()
