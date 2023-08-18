import logging
import gzip
import sys
import subprocess
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pyrodigal
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from time import process_time


def main():

    time_start = process_time()

    hmm_profiles = Path("models/hmm_profiles/baseline/baseline_profiles.hmm")
    contig = "data/raw/contigs/test_set/test1.false.contigs.fa.gz"
    #contig = "data/raw/contigs/test_set/all_seqs.true.fna.gz"
    tmp_proteins = Path(".tmp/.proteins.faa")
    tmp_result = Path(".tmp/.hmmsearch_result.txt")
    tmp_proteins.parent.mkdir(parents=True, exist_ok=True)
    pyro = pyrodigal.OrfFinder(meta=True)
    threshold = 1e-30
    number_of_contigs = 0
    positive_hits = 0

    # Load nucleotide sequence

    with gzip.open(contig, 'rt') as f:
        for contig_n, record in enumerate(SeqIO.parse(f, "fasta")):
            
            if len(record.seq) < 2500:
                continue

            number_of_contigs += 1

            # Write predicted proteins to temporary file
            with open(tmp_proteins.absolute().as_posix(), "w") as f:
                for protein_n, pred in enumerate(pyro.find_genes(bytes(record.seq))):
                    f.write(f">{contig_n+1}_{protein_n+1}\n")
                    f.write(f"{pred.translate()}\n")



            # Search protein against hmm profiles:
            cmd = ["hmmsearch", "--tblout", tmp_result.absolute().as_posix(), hmm_profiles.absolute().as_posix(), tmp_proteins.absolute().as_posix()]

            return_value = subprocess.run(cmd, stdout=subprocess.DEVNULL)
            if return_value.returncode != 0:
                logging.error(f"Error running hmmsearch on {tmp_proteins.absolute().as_posix()}")
                sys.exit()
            
            if tmp_result.exists():
                # Parse hmmsearch results:
                with open(tmp_result.absolute().as_posix(), "r") as f:
                    for line in f:
                        if line.startswith("#"):
                            continue
                        else:
                            line = line.split()
                            hmm_acc = line[0]
                            hmm_evalue = line[4]
                            hmm_score = line[5]
                            hmm_coverage = line[9]
                            if float(hmm_evalue) <= threshold:
                                positive_hits += 1
                                break
            
            # log process
            if number_of_contigs % 100 == 0:
                print(f"Contig {number_of_contigs} of 27661")
                print(f"Number of contigs: {number_of_contigs}")
                print(f"Number of positive hits: {positive_hits}")
                print(f"Percentage of positive hits: {positive_hits/number_of_contigs*100:.2f}%")
                time_end = process_time()
                print(f"Time elapsed: {time_end-time_start:.2f} seconds")
            tmp_result.unlink(missing_ok=True)
            tmp_proteins.unlink(missing_ok=True)
    
    print(f"Number of contigs: {number_of_contigs}")
    print(f"Number of positive hits: {positive_hits}")
    print(f"Percentage of positive hits: {positive_hits/number_of_contigs*100:.2f}%")

    time_end = process_time()
    print(f"Time elapsed: {time_end-time_start:.2f} seconds")
                


    # Convert predicted genes to SeqRecord objects
    # records = []
    # for gene in genes:
    #     seq = Seq(gene['translation'])
    #     print(seq)
    #     record = SeqRecord(seq, id=gene['header'], description="")
    #     records.append(record)

    # # Write SeqRecord objects to FASTA file
    # SeqIO.write(records, "predicted_proteins.fasta", "fasta")

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
