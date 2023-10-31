# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def jitter_contig_coordinates(size, start, end, reference_length):
    """ Add jitter to the contig coordinates
        and ensure it fits within the reference length.
        return adjusted coordinates within the contig.

        Returns: contig_start, contig_end, start_in_contig, end_in_contig
    """

    sat_size = end - start
    max_noise = size - sat_size
    if start - max_noise < 0:
        max_noise = start

    if max_noise < 0:
        return start, end, 0, sat_size
    
    random_noise = np.random.randint(0, max_noise)
    contig_start = start - random_noise
    contig_end = contig_start + size

    start_in_contig = random_noise
    end_in_contig = start_in_contig + sat_size

    # Adjust if contig goes beyond the reference length
    if contig_end > reference_length:
        contig_end = reference_length
        contig_start = contig_end - size

    # Adjust if contig starts before the beginning of the reference
    if contig_start < 0:
        contig_start = 0
        contig_end = contig_start + size

    start_in_contig = start - contig_start
    end_in_contig = end - contig_start

    return contig_start, contig_end, start_in_contig, end_in_contig


def sat_contig_sampling(fixed: bool = True,
                        fixed_length: int = 25000,
                        sanity_check: bool = True,
                        input_root: Path = Path("data/processed/01_combined_databases"),
                        output_root: Path = Path("data/processed/03_mock_data/satellite_contigs"),
                        sample_table_path: Path = None,
                        tmp_root: Path = None):
    """ sample reference genomes around a specific coordinate set.
        in a lognormal size distribution.
        
        Optionally do it with a fixed size.
    """

    fixed = fixed
    fixed_length = fixed_length
    sanity_check = True
    
    # Paths
    input_root = input_root
    output_root = output_root
    
    sat_seqs = Path(input_root, "all_sequences.fna")
    ref_seqs = Path(input_root, "reference_sequences")
    sat_coords = Path(input_root, "satellite_coordinates.tsv")
    tmp_fasta = Path(tmp_root, "satellite_contigs.fna")

    samples = []
    sample_coordinates = {}
    contig_coordinates = {}

    # Hardcoded parameters
    mean_log = 10.5
    sigma_log = 1.25

    # Input samples
    with open(sample_table_path, "r") as f:
        f.readline()
        for line in f:
            if line.split("\t")[2].strip() == "1":
                samples.append(line.split("\t")[0])

    # Satellite coordinates
    with open(sat_coords, "r") as f:
        for line in f:
            line = line.split("\t")
            sat_id, seq_origin_id, start, end = line[0], line[1], int(line[2]), int(line[3])
            if sat_id in samples:
                sample_coordinates[sat_id] = [seq_origin_id, start, end]

    # Generate contigs
    with open(Path(output_root, "contig_coordinates.tsv"), "w") as f_out, open(tmp_fasta, "w") as f_out_fasta:
        
        for sample_n, sat_id in enumerate(samples):
            if sample_n % 100 == 0:
                print(f"Sampling {sample_n} of {len(samples)}")

            seq_origin_id, start, end = sample_coordinates[sat_id]
            ref_path = ref_seqs / f"{seq_origin_id}.fna"

            # Load file as record:
            reference_sequence = next(SeqIO.parse(ref_path, "fasta"))

            if fixed:
                size = fixed_length
                if len(reference_sequence.seq) < size or end - start > size:
                    continue
            else:
                size = 0
                while size < end - start:
                    size = np.random.lognormal(mean_log, sigma_log)
                    size = int(np.round(size))

            contig_start, contig_end, start_in_contig, end_in_contig = jitter_contig_coordinates(
                size, start, end, len(reference_sequence.seq)
            )

            new_contig_seq = reference_sequence.seq[contig_start:contig_end]
            if new_contig_seq[start_in_contig:end_in_contig] != reference_sequence.seq[start:end]:
                print("Error: contig does not match reference sequence")
                sys.exit()

            contig_coordinates[sat_id] = [start_in_contig, end_in_contig]

            # Save satellite coordinates:
            f_out.write(f"{sat_id}\t{start_in_contig}\t{end_in_contig}\n")

            # Write contig to fasta file
            record = SeqRecord(reference_sequence.seq[contig_start:contig_end],
                               id=sat_id, description="")
            SeqIO.write(record, f_out_fasta, "fasta")

    # Sanity check input files:
    if sanity_check:
        logging.info("Loading satellite sequences for sanity check..")
        satellite_index = SeqIO.index(sat_seqs.absolute().as_posix(), "fasta")
        logging.info("Done loading satellite sequences")
        for sample_n, record in enumerate(list(SeqIO.parse(tmp_fasta, "fasta"))):
            if sample_n % 100 == 0:
                print(f"Checking sample {sample_n} of {len(samples)}")

            start, end = contig_coordinates[record.id]

            sat_seq = satellite_index[record.id].seq
            ref_seq = record.seq[start:end]

            if sat_seq != ref_seq:
                print(f"Error: {record.id} does not match {seq_origin_id} at {start}:{end}")
                print(f"{sat_seq}")
                print(f"{ref_seq}")
                sys.exit()

        satellite_index.close()

    return tmp_fasta


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    sat_contig_sampling()