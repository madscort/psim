# -*- coding: utf-8 -*-
import click
import logging
import pyhmmer
import subprocess
import sys
import pandas as pd
from pyhmmer.easel import SequenceFile
from pyhmmer.plan7 import HMMFile
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src.models.set_cover import sc_greedy
from Bio import SeqIO


# from Bio import SeqIO
# records = list(SeqIO.parse(input_filepath, "fasta"))
# print(len(records))

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

@click.command()
def main():

    rerun_all = True

    mmseq_clusters_path = Path("data/processed/protein_clusters/cluster4080DB.tsv")
    rep_protein_seqs = Path("data/processed/protein_clusters/rep4080DB.fna")

    if rep_protein_seqs.exists() and not rerun_all:
        logging.info(f"Skipping set cover, {mmseq_clusters_path.absolute().as_posix()} already exists.")
    else:
        logging.info(f"Running set cover on {mmseq_clusters_path.absolute().as_posix()}")
        ordered_cluster_reps, _ = sc_greedy(mmseq_clusters = mmseq_clusters_path)
        records = get_records_from_identifiers(identifiers=ordered_cluster_reps)
        SeqIO.write(records, rep_protein_seqs.absolute().as_posix(), "fasta")

    
    #### Pfam search ####

    pfam_hmm = Path("data/external/databases/pfam/pfam_A/Pfam-A.hmm")
    pfam_hmm_index = Path(str(pfam_hmm)+'.ssi')
    
    # hmm search parameters:
    e_cutoff = "1e-3"
    p_cpus = "6"
    hmmsearch_out = Path("data/processed/protein_clusters/pfam_rep4080DB_res.txt")
    hmmsearch_parsed_results = Path("data/processed/protein_clusters/pfam_rep4080DB_res_parsed.tsv")
    hmmsearch_accessions = Path("data/processed/protein_clusters/pfam_rep4080DB_res_parsed_unique_acc.tsv")
    
    hmmfetch_profiles = Path("models/hmm_profiles/baseline/baseline_profiles.hmm")

    # Index the pfam database:

    if pfam_hmm_index.exists() and not rerun_all:
        logging.info(f"Skipping hmm index, {pfam_hmm_index.absolute().as_posix()} already exists.")
    else:
        pfam_hmm_index.unlink(missing_ok=True)

        logging.info(f"Indexing {pfam_hmm.absolute().as_posix()}")
        cmd = ['hmmfetch', '--index', pfam_hmm.absolute().as_posix()]
        return_out = subprocess.run(cmd, stdout=subprocess.DEVNULL)

        if return_out.returncode != 0:
            raise Exception(f"Error running hmmfetch: {return_out}")


    # Run hmmsearch on the pfam database:

    if hmmsearch_out.exists() and not rerun_all:
        logging.info(f"Skipping hmmsearch, {hmmsearch_out.absolute().as_posix()} already exists.")
    else:
        hmmsearch_out.unlink(missing_ok=True)

        logging.info(f"Running hmmsearch on {rep_protein_seqs.absolute().as_posix()}")
        
        cmd = f"hmmsearch --acc --tblout {hmmsearch_out.absolute().as_posix()} --cpu {p_cpus} -E {e_cutoff} {pfam_hmm.absolute().as_posix()} {rep_protein_seqs.absolute().as_posix()}"
        subprocess_return = subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL)
        if subprocess_return.returncode != 0:
            raise Exception(f"Error running hmmsearch: {subprocess_return}")
    

    # Parse the hmmsearch results, only take first four columns:

    if hmmsearch_parsed_results.exists() and not rerun_all:
        logging.info(f"Skipping hmmsearch parsing, {hmmsearch_parsed_results.absolute().as_posix()} already exists.")
    else:

        hmmsearch_parsed_results.unlink(missing_ok=True)
        hmmsearch_accessions.unlink(missing_ok=True)

        logging.info(f"Parsing hmmsearch results from {hmmsearch_out.absolute().as_posix()}")

        hmm_df = pd.DataFrame(columns=["protein", "pfam", "accession", "e_value", "score"])

        grep_cmd = ["grep", "-vE", '^#', f"{hmmsearch_out.absolute().as_posix()}"]
        grep_out = subprocess.Popen(grep_cmd, stdout=subprocess.PIPE)
        parsed_out = subprocess.check_output(["awk", "-v", "OFS='\t'", '{print $1, $3, $4, $5, $6}'], stdin=grep_out.stdout)
        for line in parsed_out.splitlines():
            parsed_col = line.decode("utf-8").replace("'","").split("\t")
            hmm_df = pd.concat([hmm_df, 
                                pd.DataFrame([parsed_col],
                                            columns=["protein", "pfam", "accession", "e_value", "score"])],
                                            ignore_index=True)

        # For each protein, get the best hit and unique pfams:
        best_hits = hmm_df.groupby("protein")["e_value"].min().reset_index()
        best_hits = best_hits.merge(hmm_df, on=["protein", "e_value"], how="left")
        
        logging.info(f"Number of proteins with hits: {len(best_hits)}")
        logging.info(f"Number of unique pfams: {len(best_hits['pfam'].unique())}")
        
        # # For each pfam with multiple proteins, output the name of the pfam and proteins:
        # pfam_counts = best_hits.groupby("pfam")["protein"].count().reset_index()
        # print(pfam_counts[pfam_counts["protein"] > 1])

        # Print protein id and accession for all hits to tsv:
        best_hits[["protein", "e_value", "accession"]].to_csv(hmmsearch_parsed_results.absolute().as_posix(), sep="\t", index=False)
        # Print unique accession numbers only to tsv:
        best_hits[["accession"]].drop_duplicates().to_csv(hmmsearch_accessions.absolute().as_posix(), sep="\t", index=False, header=False)

    # Get hmms from pfam database and press to binary:

    if hmmfetch_profiles.exists() and not rerun_all:
        logging.info(f"Skipping hmmpfetch and press, {hmmfetch_profiles.absolute().as_posix()} already exists.")
    else:
        for ps in hmmfetch_profiles.parent.glob(str(hmmfetch_profiles.name) + "*"):
            ps.unlink(missing_ok=True)

        if not hmmfetch_profiles.parent.exists():
            hmmfetch_profiles.parent.mkdir(parents=True)

        cmd = ['hmmfetch','-o', hmmfetch_profiles.absolute().as_posix(), '-f', pfam_hmm.absolute().as_posix(), hmmsearch_accessions.absolute().as_posix()]
        return_out = subprocess.run(cmd,
                                    stdout=subprocess.DEVNULL)
        if return_out.returncode != 0:
            raise Exception(f"Error running hmmfetch: {return_out}")

        cmd = ['hmmpress', hmmfetch_profiles.absolute().as_posix()]
        return_out = subprocess.run(cmd,
                                    stdout=subprocess.DEVNULL)
        if return_out.returncode != 0:
            raise Exception(f"Error running hmmpress: {return_out}")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
