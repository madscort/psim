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


def sat_contig_sampling(fixed: bool = False, fixed_length: int = 25000,
                        sanity_check: bool = True,
                        input_root: Path = Path("data/processed/01_combined_renamed"),
                        output_root: Path = Path("data/processed/03_mock_data/satellite_contigs")):
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
    ref_seqs = Path(input_root, "all_reference_sequences.fna")
    sat_coords = Path(input_root, "satellite_coordinates.tsv")
    sample_table = pd.read_csv(Path(input_root, "sample_table.tsv"),
                               sep="\t", header=None, names=['id', 'type', 'label'])
    

    # Hardcoded parameters
    mean_log = 10.5
    sigma_log = 1.25

    reference_index = SeqIO.index(ref_seqs.absolute().as_posix(), "fasta")

    # Number of samples to generate
    with open(sat_coords, "r") as f:
        n_samples = sum(1 for line in f)
    
    # Generate contigs
    with open(sat_coords, "r") as f, open(Path(output_root, "sat_contigs_coord.tsv"), "w") as f_out, open(Path(output_root, "sample_table.tsv"), 'w') as f_out_table, open(Path(output_root, "sat_contigs.fna"), "w") as f_out_fasta:
        
        for line_n, line in enumerate(f):
            if line_n % 100 == 0:
                print(f"Sampling {line_n} of {n_samples}")
            line = line.split("\t")
            
            sat_id, seq_origin_id, start, end = line[0], line[1], int(line[2]), int(line[3])

            if fixed:
                size = fixed_length
                if len(reference_index[seq_origin_id].seq) < size or end - start > size:
                    continue
            else:
                size = 0
                while size < end - start:
                    size = np.random.lognormal(mean_log, sigma_log)
                    size = int(np.round(size))

            contig_start, contig_end, start_in_contig, end_in_contig = jitter_contig_coordinates(
                size, start, end, len(reference_index[seq_origin_id].seq)
            )

            new_contig_seq = reference_index[seq_origin_id].seq[contig_start:contig_end]
            if new_contig_seq[start_in_contig:end_in_contig] != reference_index[seq_origin_id].seq[start:end]:
                print("Error: contig does not match reference sequence")
                sys.exit()

            # Save satellite coordinates:
            f_out.write(f"{sat_id}\t{start_in_contig}\t{end_in_contig}\n")

            # Save new redacted sample table:
            f_out_table.write(f"{sat_id}\t{sample_table[sample_table['id'] == sat_id]['type'].values[0]}\t{sample_table[sample_table['id'] == sat_id]['label'].values[0]}\n")

            # Write contig to fasta file
            record = SeqRecord(reference_index[seq_origin_id].seq[contig_start:contig_end],
                               id=sat_id, description="")
            SeqIO.write(record, f_out_fasta, "fasta")

    # Sanity check input files:
    if sanity_check:
        with open(sat_coords, "r") as f:
            
            satellite_index = SeqIO.index(sat_seqs.absolute().as_posix(), "fasta")
            
            for line_n, line in enumerate(f):
                if line_n % 100 == 0:
                    print(f"Checking line {line_n} of {n_samples}")
                line = line.split("\t")
                sat_id, seq_origin_id, start, end = line[0], line[1], int(line[2]), int(line[3])

                sat_seq = satellite_index[sat_id].seq
                ref_seq = reference_index[seq_origin_id].seq[start:end]
                if sat_seq != ref_seq:
                    print(f"Error: {sat_id} does not match {seq_origin_id} at {start}:{end}")
                    print(f"{sat_seq}")
                    print(f"{ref_seq}")
                    sys.exit()
    
        satellite_index.close()
    
    reference_index.close()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    sat_contig_sampling()