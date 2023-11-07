# -*- coding: utf-8 -*-
import click
import logging
import sys
import gzip
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

def get_seq_count_fasta(file_path):
    with open(file_path, 'r') as f:
        return sum(1 for line in f if line.startswith('>'))
def get_non_empty_line_count(file_path):
    with open(file_path, 'r') as f:
        return sum(1 for line in f if line.strip())


@click.command()
def main():
    """ Rename all satellite genomes and their proteins.
        Create mapping table with old and new sequence identifiers.
        Create table with sequence identifier, reference genome and coordinates.
        Create table with sequence identifier, type and label.
    """
    # Input files and metadata

    meta_urvish = Path(project_dir, "data/raw/Dataset_Micheal/RefSeq_PICI_table.tsv")
    protein_root_urvish = Path(project_dir, "data/raw/Dataset_Micheal/06_michael_satellites_gene_prediction/")
    satellite_sequences_urvish = Path(project_dir, "data/raw/Dataset_Micheal/RefSeq_ALL_PICIs.fasta")
    reference_genomes_urvish = Path(project_dir, "data/raw/Dataset_Micheal/RefSeq_ALL_PICIs_host_genomes.fasta")

    meta_rocha = Path(project_dir, "data/raw/Dataset_Roche/satellites_coordinates_ongenome.tsv")
    protein_root_rocha = Path(project_dir, "data/raw/Dataset_Roche/File_S4_SatelliteProteomes/")
    satellite_sequences_root_rocha = Path(project_dir, "data/raw/Dataset_Roche/")
    reference_genomes_root_rocha = Path(project_dir, "data/raw/Dataset_Roche/host_genomes/")

    # Output files

    mapping_table = Path(project_dir, "data/processed/01_combined_databases/renaming_map.tsv")
    satellite_coord = Path(project_dir, "data/processed/01_combined_databases/satellite_coordinates.tsv")
    sample_table = Path(project_dir, "data/processed/01_combined_databases/sample_table.tsv")
    all_sequences = Path(project_dir, "data/processed/01_combined_databases/all_sequences.fna")
    all_proteins = Path(project_dir, "data/processed/01_combined_databases/all_proteins.faa")
    all_reference_sequences = Path(project_dir, "data/processed/01_combined_databases/all_reference_sequences.fna")
    tmp_genomes_rocha = Path(project_dir, "data/processed/01_combined_databases/tmp_genomes_rocha.fna")
    
    # Urvish manual update of protein files:
    new_protein_id = {
        "NZ_CP213122662131.1": "NZ_CP022660.1",
        "NZ_CP213545213554.1": "NZ_CP045054.1",
        "NZ_CP213545213558.1": "NZ_CP045058.1",
        "NZ_CP213445213461.1": "NZ_CP045061.1",
        "NZ_CP675176751523.1": "NZ_CP070553.1",
        "NZ_CP403212119.2": "NZ_CP012119.2",
        "NZ_CP178374117837117837.1": "NZ_CP041010.1",
        "NZ_CP263745472637.1": "NZ_CP045470.1",
        "NZ_CP142245471.1": "NZ_CP045471.1",
        "NZ_CP438745469.1": "NZ_CP045469.1",
        "NZ_CP183754444.1": "NZ_CP054444.1",
        "NZ_CP335769874.1": "NZ_CP069874.1"
    }

    multi_copy_set = set()
    multi_copy_set_urvish = set()
    multi_copy_set_rocha = set()
    urid_set = set()
    origins_set = set()
    sat_dict = {}
    ref_ids = set()
    total_ps_count = 0
    ps_count = 0
    total_protein_count = 0
    skipped = 0

    # Load satellite sequences to memory.
    # This is necessary because of the senseless headers.
    with open(satellite_sequences_urvish, "r") as f:
        for record in SeqIO.parse(f, "fasta"):
            origin = record.description.split(";")[0].split(" ")[0]
            origins_set.add(origin)
            start = record.description.split(";")[3]
            end = record.description.split(";")[4]
            urid = f"{origin}_{start}_{end}"
            sat_dict[urid] = record.seq
            urid_set.add(urid)

    # Get all reference genome sequence ids:
    with open(reference_genomes_urvish, "r") as f:
        ref_ids = {record.id.replace("*"," ").split(" ")[0].strip() for record in SeqIO.parse(f, "fasta")}

    # Index reference genomes for fast lookup.
    ref_record_index = SeqIO.index(str(reference_genomes_urvish), "fasta")

    # Urvish
    logging.info(f"Reading data from {meta_urvish.name}")

    with open(meta_urvish, "r") as f, open(mapping_table, "w") as map_f, open(satellite_coord, "w") as coord_f, open(sample_table, "w") as sample_f, open(all_proteins, "w") as proteins_f, open(all_sequences, "w") as seq_f, open(all_reference_sequences, "w") as ref_f:
        urid_coord_set = set()
        for line_n, line in enumerate(f):
            if line_n % 1000 == 0:
                logging.info(f"Processed {line_n} Urvish satellites.")
            line = line.split("\t")
            line = [x.strip() for x in line]
            ref_origin_seq, type, ref_genome, start, end = line[0].replace("*"," ").split(" ")[0], line[1].upper(), line[0], line[3], line[4]
            urid_coord = f"{ref_origin_seq}_{start}_{end}"
            multi = False

            # Skip if protein id is in manual updated list, todo:
            if ref_origin_seq in new_protein_id.keys():
                logging.info(f"Skipping {ref_origin_seq} because of inconsistent phenotype.")
                skipped += 1
                continue
        
            # Check if duplicate:
            if urid_coord in urid_coord_set:
                logging.info(f"Duplicate urvish id {urid_coord}. Skipping.")
                skipped += 1
                continue

            # Check if genome exists:
            if urid_coord not in urid_set:
                logging.info(f"Error finding satellite sequence for {ref_origin_seq} : {start} {end}\n{ref_genome}")
                skipped += 1
                continue
            
            # Check if protein sequence exists:
            if len(list(Path(protein_root_urvish).glob(f"{ref_origin_seq}*{start}_{end}.faa"))) != 1:
                logging.info(f"Error finding protein sequence for {ref_origin_seq} : {start} {end}\n{ref_genome}")
                skipped += 1
                continue

            # Check if reference genome exists:
            if ref_origin_seq not in ref_ids:
                logging.info(f"Error finding reference genome sequence for {ref_origin_seq} : {start} {end}\n{ref_genome}")
                skipped += 1
                continue
            
            # Check if satellite sequence matches reference genome sequence:
            ref_seq = ref_record_index[ref_origin_seq].seq
            if sat_dict[urid_coord] != ref_seq[int(start):int(end)]:
                logging.info(f"Error: satellite sequence does not match reference genome sequence for {ref_origin_seq} : {start} {end}\n{ref_genome}")
                skipped += 1
                continue

            # Check if multi-copy:
            if ref_origin_seq in multi_copy_set:
                multi = True
            multi_copy_set.add(ref_origin_seq)
            multi_copy_set_urvish.add(ref_origin_seq)

            # All checks passed:
            ps_count += 1
            urid_coord_set.add(urid_coord)
            new_id = f"PS_U{ps_count}"

            map_f.write(f"{new_id}\t{ref_genome}\n")
            coord_f.write(f"{new_id}\t{ref_origin_seq}\t{start}\t{end}\n")
            sample_f.write(f"{new_id}\t{type}\t1\n")
            
            # Rename proteins: 
            with open(list(Path(protein_root_urvish).glob(f"{ref_origin_seq}*{end}.faa"))[0], "r") as pf:
                for protein_n, record in enumerate(SeqIO.parse(pf, "fasta")):
                    record.id = f"{new_id}|PROTEIN_{protein_n+1}"
                    record.description = ""
                    SeqIO.write(record, proteins_f, "fasta")
                total_protein_count += protein_n + 1
            
            # Create new satellite sequence record and add to combined output:
            record = SeqRecord(Seq(sat_dict[urid_coord]),
                               id=new_id,
                               description="")
            SeqIO.write(record, seq_f, "fasta")

            # Add reference sequence to combined output, only if not multi-copy:
            if not multi:
                record = SeqRecord(ref_record_index[ref_origin_seq].seq,
                                id=ref_record_index[ref_origin_seq].id,
                                description=ref_record_index[ref_origin_seq].description)
                SeqIO.write(record, ref_f, "fasta")
    
    ref_record_index.close()
    total_ps_count += ps_count
    ps_count = 0

    logging.info(f"Number of genomes in Urvish data: {total_ps_count}")
    logging.info(f"Number of proteins in Urvish data: {total_protein_count}")
    logging.info(f"Number of skipped satellites: {skipped}")

    # Rocha
    
    rocha_ids = set()
    rocha_ref_ids = set()
    logging.info(f"Reading data from {meta_rocha.name}")

    # Combine all rocha genomes and index for fast lookup:
    if len(list(satellite_sequences_root_rocha.glob("*.fst"))) != 4:
        raise Exception(f"Not correct number of satellite sequence types")
    else:
        tmp_genomes_rocha.unlink(missing_ok=True)
        for type_file in satellite_sequences_root_rocha.glob("*.fst"):
            with open(type_file, "r") as f, open(tmp_genomes_rocha, "a") as gen_f:
                for record in SeqIO.parse(f, "fasta"):
                    record.id = record.id.strip()
                    rocha_ref_ids.add(record.id)
                    SeqIO.write(record, gen_f, "fasta")
        rocha_genome_index = SeqIO.index(tmp_genomes_rocha.absolute().as_posix(), "fasta")

    with open(meta_rocha, "r") as f, open(mapping_table, "a") as map_f, open(satellite_coord, "a") as coord_f, open(sample_table, "a") as sample_f, open(all_proteins, "a") as proteins_f, open(all_sequences, "a") as seq_f, open(all_reference_sequences, "a") as ref_f:

        next(f) # skip header

        for line_n, line in enumerate(f):
            if line_n % 1000 == 0:
                logging.info(f"Processed {line_n} Rocha satellites.")
            line = line.split("\t")
            line = [x.strip() for x in line]
            multi = False
            old_id, type, ref_genome, ref_seq_id, start, end = line[0], line[0].split(".")[4], line[1], line[2], int(line[3])-1, line[4] # -1 because of wrong indexing

            roid = f"{old_id}_{start}_{end}"

            # Check if duplicate:
            if roid in rocha_ids:
                print(f"Duplicate Rocha genome id {old_id} - {start} {end}.")
                continue
            rocha_ids.add(roid)

            # Check if multi-copy:
            if ref_seq_id in multi_copy_set:
                multi = True
            multi_copy_set.add(ref_seq_id)
            multi_copy_set_rocha.add(ref_seq_id)
            
            # Check if genome exists:
            if old_id not in rocha_ref_ids:
                print(f"Error finding satellite sequence for {old_id} : {start} {end}\n{ref_genome}")
                skipped += 1
                continue
            
            # Check if reference genome exists:
            if len(list(reference_genomes_root_rocha.glob(f"{ref_genome}*.fna"))) != 1:
                raise Exception(f"Not correct number of satellite sequence types: {ref_genome}")
            else:
                ref_genome_index = SeqIO.index(list(reference_genomes_root_rocha.glob(f"{ref_genome}*.fna"))[0].absolute().as_posix(), "fasta")
                try:
                    ref_genome_index[ref_seq_id]
                except KeyError:
                    print(f"Error finding reference genome sequence for {old_id} : {start} {end}\n{ref_genome}")
                    skipped += 1
                    continue
            
            # Check if satellite sequence matches reference genome sequence:
            ref_seq = ref_genome_index[ref_seq_id].seq
            if rocha_genome_index[old_id].seq != ref_seq[int(start):int(end)]:
                print(f"Error: satellite sequence does not match reference genome sequence for {old_id} : {start} {end}\n{ref_genome}")
                print(rocha_genome_index[old_id].seq[:100])
                print(ref_seq[int(start)-1:int(end)][:100])
                skipped += 1
                continue

            # All checks passed:
            ps_count += 1
            new_id = f"PS_R{ps_count}"

            map_f.write(f"{new_id}\t{old_id}\n")
            coord_f.write(f"{new_id}\t{ref_seq_id}\t{start}\t{end}\n")
            sample_f.write(f"{new_id}\t{type}\t1\n")

            # Rename and add proteins:
            with open(list(Path(protein_root_rocha).rglob(f"{old_id}.prt"))[0], "r") as pf:
                for protein_n, record in enumerate(SeqIO.parse(pf, "fasta")):
                    record.id = f"{new_id}|PROTEIN_{protein_n+1}"
                    record.description = ""
                    SeqIO.write(record, proteins_f, "fasta")
                total_protein_count += protein_n + 1

            # Create new satellite sequence record and add to combined output:
            record = SeqRecord(Seq(rocha_genome_index[old_id].seq),
                               id=new_id,
                               description="")
            SeqIO.write(record, seq_f, "fasta")

            # Add reference sequence to combined output, if not multi-copy:
            if not multi:
                record = SeqRecord(ref_genome_index[ref_seq_id].seq,
                                id=ref_genome_index[ref_seq_id].id,
                                description=ref_genome_index[ref_seq_id].description)
                SeqIO.write(record, ref_f, "fasta")

    total_ps_count += ps_count

    logging.info(f"Number of satellites in data: {total_ps_count}")
    logging.info(f"Number of skipped satellites: {skipped}")
    logging.info(f"Number of proteins in data: {total_protein_count}")
    logging.info(f"Intersection of Urvish and Rocha ref genomes: {len(multi_copy_set_urvish.intersection(multi_copy_set_rocha))}")

    ref_genome_index.close()        
    rocha_genome_index.close()
    tmp_genomes_rocha.unlink(missing_ok=True)

    logging.info(f"Done!")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
