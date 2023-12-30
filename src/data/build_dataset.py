# -*- coding: utf-8 -*-
import click
import logging
import subprocess
import sys
import gzip
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from src.data.sat_contig_sampling import sat_contig_sampling
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from Bio import SeqIO
from tempfile import TemporaryDirectory
from src.data.vir_contig_sampling import fixed_length_viral_sampling

def get_provirus_taxonomy():
    img_vr_host_info = Path("data/external/databases/IMG_VR/IMGVR_all_Host_information-high_confidence.tsv")
    img_seq_species_dict = {}

    with open(img_vr_host_info, "r") as f:
        f.readline()
        for line in f:
            sequence = line.split("\t")[0].strip()
            host = line.split("\t")[2].split(";")
            family, genus, species = host[4][3:].strip(), host[5][3:].strip(), host[6][3:].strip()
            if family == "" and genus == "" and species == "":
                continue
            else:
                img_seq_species_dict[sequence] = "_".join(species.split(" ")).lower()
        return img_seq_species_dict

def get_host_taxonomy():
    taxonomy = Path("data/processed/01_combined_databases/ps_tax_info.tsv")
    coord = Path("data/processed/01_combined_databases/satellite_coordinates.tsv")
    ref_tax_dict = {}
    sample_ref_dict = {}
    with open(coord, "r") as f:
        for line in f:
            sample_ref_dict[line.split("\t")[0].strip()] = line.split("\t")[1].strip()
    
    with open(taxonomy, "r") as f:
        for line in f:
            sample = line.strip().split("\t")
            try:
                ref  = sample_ref_dict[sample[0].strip()]
            except KeyError:
                print(f"Sample {sample[0]} not found in satellite coordinates file!")
                continue
            try:
                ref_tax_dict[ref] = "_".join(sample[3].lower().split(" ")).strip()
            except IndexError:
                print(line)
                ref_tax_dict[ref] = "unknown"
    return ref_tax_dict


def create_dataset():
    input = Path("data/processed/01_combined_databases")
    root = Path("data/processed/10_datasets")
    dataset_id = "dataset_v02"
    dataset_root = Path(root, dataset_id)

    length = 25000

    sampletable = Path("data/processed/02_preprocessed_database/02_homology_reduction/sampletable.tsv")
    host_input = Path("data/processed/06_host_sequences/sampled_host_seqs.90_DOWNSAMPLED.fna")
    viral_input = Path("data/processed/05_viral_sequences/imgvr_filtered/cd.hit.IMGVR_minimal_seqs.90_DOWNSAMPLED.fna")
    meta_input = Path("data/processed/04_metagenomic_contigs/almeida/almeida_rand.90_DOWNSAMPLED.fna")

    sequence_collection = {'host': host_input,
                           'provirus': viral_input,
                           'meta': meta_input}

    provirus_tax = get_provirus_taxonomy()
    host_tax = get_host_taxonomy()

    sampletable_values = []
    # Create list of lists with id, type and labels from sampletable:
    with open(sampletable, "r") as f:
        for line in f:
            line = line.strip().split("\t")
            id, type, label = line[0], line[1], int(line[2])
            sampletable_values.append([id, type, label])
    
    type_list = []
    # Add negative samples
    for seq_type in sequence_collection:
        with open(sequence_collection[seq_type], "r") as f:
            records = list(SeqIO.parse(f, "fasta"))
            for record in records:
                if seq_type == "provirus":
                    try:
                        type_id = f"provirus_{provirus_tax[record.id]}"
                    except KeyError:
                        type_id = "provirus_unknown"
                elif seq_type == "host":
                    try:
                        type_id = f"host_{host_tax['_'.join(record.id.split('_')[:-2])]}"
                    except KeyError:
                        print(record.id)
                        sys.exit()
                elif seq_type == "meta":
                    type_id = "meta"
                type_list.append(type_id)
                sampletable_values.append([record.id, type_id, 0])

    # Prevent failed stratification by renaming types with less than 4 samples (we need two times two splits) to singletype:
    cnt = dict(Counter(type_list))
    singletypes = []
    for key in cnt:
        if cnt[key] < 2:
            singletypes.append(key)
    sample_type_dict = {}
    
    for n, value in enumerate(sampletable_values):
        if value[1] in singletypes:
            sample_type_dict[value[0]] = value[1]
            sampletable_values[n][1] = "singletype"

    # Read as dataframe:
    df = pd.DataFrame(sampletable_values, columns=['id', 'type', 'label'])
    
    # Make training, validation and test split (80/10/10):
    train, non_train = train_test_split(df, stratify=df['type'], test_size=0.2, random_state=1)
    
    # Repeat stratification fail prevention:
    non_train_types = list(non_train['type'])
    cnt = dict(Counter(non_train_types))
    for key in cnt:
        if cnt[key] < 2:
            singletypes.append(key)
            id = non_train.loc[non_train['type'] == key, 'id'].values
            sample_type_dict[id[0]] = key

    for index, row in non_train.iterrows():
        if row['type'] in singletypes:
            non_train.at[index, 'type'] = "singletype"

    test, val = train_test_split(non_train, stratify=non_train['type'], test_size=0.5, random_state=1)

    # Add type type ids back to sampletables:
    for df in [train, val, test]:
        for index, row in df.iterrows():
            if row['id'] in sample_type_dict:
                df.at[index, 'type'] = sample_type_dict[row['id']]

    # Get positive size:
    logging.info(f"Positive size - Train: {train[train['label'] == 1].shape[0]}, Validation: {val[val['label'] == 1].shape[0]}, Test: {test[test['label'] == 1].shape[0]}")
    logging.info(f"Negative size - Train: {train[train['label'] == 0].shape[0]}, Validation: {val[val['label'] == 0].shape[0]}, Test: {test[test['label'] == 0].shape[0]}")

    # Create dataset folder:
    dataset_root.mkdir(parents=True, exist_ok=True)

    # Create sampletables
    train_path = Path(dataset_root, "train.tsv")
    val_path = Path(dataset_root, "val.tsv")
    test_path = Path(dataset_root, "test.tsv")
    train.to_csv(train_path, sep="\t", index=False)
    val.to_csv(val_path, sep="\t", index=False)
    test.to_csv(test_path, sep="\t", index=False)

    data_splits = {'train': train_path,
                    'val': val_path,
                    'test': test_path}

    # Create individual folders with data splits:
    for split in data_splits:
        with TemporaryDirectory() as tmp:
            
            tmp = Path(tmp)
            split_samples = []
            with open(data_splits[split], "r") as f:
                f.readline()
                for line in f:
                    split_samples.append(line.split("\t")[0])
            
            # Create split folder:
            split_path = Path(dataset_root, split)
            split_path.mkdir(parents=True, exist_ok=True)

            # Get sequence window around satellite coordinates:
            satellite_contigs = sat_contig_sampling(sample_table_path=data_splits[split],
                                                    fixed=True,
                                                    fixed_length=length,
                                                    sanity_check=True,
                                                    input_root=input,
                                                    output_root=split_path,
                                                    tmp_root=tmp)

            # Get split host sequences:
            host_contigs = Path(tmp, "host_contigs.fna")
            with open(host_input, "r") as f_in, open(host_contigs, "w") as f_out:
                for record in SeqIO.parse(f_in, "fasta"):
                    if record.id in split_samples:
                        SeqIO.write([record], f_out, "fasta")

            # Get split metagenomic sequences:
            meta_contigs = Path(tmp, "meta_contigs.fna")
            with open(meta_input, "r") as f_in, open(meta_contigs, "w") as f_out:
                for record in SeqIO.parse(f_in, "fasta"):
                    if record.id in split_samples:
                        SeqIO.write([record], f_out, "fasta")
            
            # Get split viral sequences:
            viral_split_fasta = Path(tmp, "viral_split.fna")
            with open(viral_input, "r") as f_in, open(viral_split_fasta, "w") as f_out:
                for record in SeqIO.parse(f_in, "fasta"):
                    if record.id in split_samples:
                        SeqIO.write([record], f_out, "fasta")

            # Get sequence window in provirus sequences:
            viral_contigs = fixed_length_viral_sampling(number = 0,
                                                        length=length,
                                                        input_fasta_path=viral_split_fasta,
                                                        output_root=tmp)
            
            # Final sequence collection:
            contig_collection = {'satellite_contigs': satellite_contigs,
                                 'host_contigs': host_contigs,
                                 'meta_contigs': meta_contigs,
                                 'viral_contigs': viral_contigs}
            
            # Collect all sequences in one file:
            seq_ids = []
            with open(split_path / "sequences.fna", "w") as out_f:
                for fasta in contig_collection.values():
                    with open(fasta, "r") as in_f:
                        for line in in_f:
                            out_f.write(line)
                            if line.startswith(">"):
                                seq_ids.append(line.split()[0][1:])

            # Create a folder with a file per sequence:
            sequences = split_path / "sequences"
            sequences.mkdir(parents=True, exist_ok=True)
            
            for record in SeqIO.parse(split_path / "sequences.fna", "fasta"):
                with open(Path(sequences, f"{record.id}.fna"), "w") as out_f:
                    SeqIO.write(record, out_f, "fasta")

            # Create a folder with a file per raw satellite sequence:
            sat_sequences = Path(split_path, "satellite_sequences")
            sat_sequences.mkdir(parents=True, exist_ok=True)

            for record in SeqIO.parse(Path(input) / "all_sequences.fna", "fasta"):
                if record.id in split_samples:
                    with open(Path(sat_sequences, f"{record.id}.fna"), "w") as out_f:
                        SeqIO.write(record, out_f, "fasta")

            print("Sanity check:", split)
            print("Number of samples: ", len(split_samples))
            if len(split_samples) != len(set(split_samples)):
                print("Duplicate samples")
            print("Number of sequences: ", len(seq_ids))
            if len(seq_ids) != len(set(seq_ids)):
                print("Duplicate sequences")

            # Not all sequences end up in dataset in the case of fixed sequence lengths
            # sometimes flanking regions are too short for sampling the sequences for example.
            # The sampletable is updated here:

            sample_update = []
            for sample in split_samples:
                if sample not in seq_ids:
                    print(sample, " not in seqs")
                    sample_update.append(sample)
            for seq in seq_ids:
                if seq not in split_samples:
                    print(seq, " not in samples")

            if len(sample_update) != 0:
                print("Updating sampletable")
                with open(data_splits[split], "r") as f_in, open(tmp / "tmp_table.tsv", "w") as f_out:
                    for line in f_in:
                        if line.split("\t")[0] not in sample_update:
                            print(line.strip(), file=f_out)
                with open(data_splits[split], "w") as f_out, open(tmp / "tmp_table.tsv", "r") as f_in:
                    for line in f_in:
                        print(line.strip(), file=f_out)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    create_dataset()