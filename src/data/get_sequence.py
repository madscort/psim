import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pyrodigal
from Bio import SeqIO

def get_gene_gc_sequence(fna_file: Path = Path("data/processed/10_datasets/phage_25_reduced_90/sequences/PS_U370.fna")):
    """ Takes a fasta file and returns a list of GC contents for each gene."""

    input_sequence_file = fna_file
    output = list()
    pyro = pyrodigal.GeneFinder(meta=True)
    
    with open(input_sequence_file, 'r') as f:
        contig = SeqIO.parse(f, "fasta").__next__()
        for n, gene in enumerate(pyro.find_genes(bytes(contig.seq))):
            output.append(gene.gc_cont)
    return output

def get_protein_seq(fna_file: Path = Path("data/processed/10_datasets/phage_25_reduced_90/sequences/PS_U370.fna")):
    """ Takes a fna file and returns seq object of proteins."""

    input_sequence_file = fna_file
    output = list()
    pyro = pyrodigal.GeneFinder(meta=True)
    
    with open(input_sequence_file, 'r') as f:
        contig = SeqIO.parse(f, "fasta").__next__()
        for n, gene in enumerate(pyro.find_genes(bytes(contig.seq))):
            output.append(gene.translate())
    return output

def get_clusters(fna_file: Path = Path("data/processed/10_datasets/phage_25_reduced_90/sequences/PS_U370.fna")):
    """ Takes a fasta file and returns a list of GC contents for each gene."""

    input_sequence_file = fna_file
    output = list()
    pyro = pyrodigal.GeneFinder(meta=True)
    
    with open(input_sequence_file, 'r') as f:
        contig = SeqIO.parse(f, "fasta").__next__()
        for n, gene in enumerate(pyro.find_genes(bytes(contig.seq))):
            output.append(gene.gc_cont)
    return output

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    get_protein_seq()