# -*- coding: utf-8 -*-
import click
import logging
import subprocess
import sys
import pandas as pd
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from Bio import SeqIO

def get_hmms(rerun: bool = False,
             pfamA: bool = True,
             pfam_hmm: Path = Path("data/external/databases/pfam/pfam_A/Pfam-A.hmm"),
             re_index_pfam: bool = False,
             rep_protein_seqs: Path = Path("data/processed/protein_clusters/test/rep3090DB.fna"),
             out_prefix: str = "cluster3090",
             out_file_root: Path = Path("data/processed/protein_clusters/pfam/cluster3090DB")):

    rerun_all = rerun
    re_index = re_index_pfam
    pfamA = pfamA
    rep_protein_seqs = rep_protein_seqs

    #### Pfam search ####

    pfam_hmm = pfam_hmm
    pfam_hmm_index = Path(str(pfam_hmm)+'.ssi')
    
    # hmm search parameters:
    e_cutoff = "1e-3"
    p_cpus = "6"

    hmmsearch_out = Path(out_file_root, f"{out_prefix}_pfam_res.txt")
    hmmsearch_parsed_results = Path(out_file_root, f"{out_prefix}_pfam_res_parsed.tsv")
    hmmsearch_accessions = Path(out_file_root, f"{out_prefix}_pfam_res_parsed_unique_acc.tsv")
    hmmfetch_profiles = Path(out_file_root, f"{out_prefix}_hmm_profiles.hmm")

    hmmsearch_out.parent.mkdir(parents=True, exist_ok=True)

    # Index the pfam database:

    if pfam_hmm_index.exists() and not re_index:
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
        if pfamA:
            parsed_out = subprocess.check_output(["awk", "-v", "OFS='\t'", '{print $1, $3, $4, $5, $6}'], stdin=grep_out.stdout)
        else:
            parsed_out = subprocess.check_output(["awk", "-v", "OFS='\t'", '{print $1, $4, $3, $5, $6}'], stdin=grep_out.stdout)
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
        pfam_counts = best_hits.groupby("pfam")["protein"].count().reset_index()
        print(pfam_counts[pfam_counts["protein"] > 1])

        # Print protein id and accession for all hits to tsv:

        best_hits[["protein", "e_value", "accession"]].to_csv(hmmsearch_parsed_results.absolute().as_posix(), sep="\t", index=False)
        # Print unique accession numbers only to tsv:
        best_hits[["accession"]].drop_duplicates().to_csv(hmmsearch_accessions.absolute().as_posix(), sep="\t", index=False, header=False)

    # Get hmms from pfam database and press to binary:

    if hmmfetch_profiles.exists() and not rerun_all:
        logging.info(f"Skipping hmmfetch and press, {hmmfetch_profiles.absolute().as_posix()} already exists.")
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

    return hmmfetch_profiles

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    get_hmms()