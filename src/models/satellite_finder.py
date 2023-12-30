from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
from typing import List, Tuple
from tempfile import TemporaryDirectory
import logging
import sys
import pandas as pd
from Bio import SeqIO
from Bio.SeqIO.FastaIO import SimpleFastaParser
import pyrodigal_gv
import collections
import re
from src.models.LNSequenceModule import SequenceModule
from src.data.LN_data_module import encode_sequence
from src.data.build_stringDB_pfama import get_one_string
from torch.nn.functional import softmax
from src.data.build_stringDB_novo import HMMER
from src.data.build_stringDB_blitsDB import SMMseqs2
import subprocess

translate = str.maketrans("ACGTURYKMSWBDHVN", "0123444444444444")
Protein = collections.namedtuple("Protein", ["accession", "contig", "contig_num", "contig_order", "start", "end", "contig_len"])
Sequence = collections.namedtuple("Sequence", ["id", "host", "seq"])
Hit = collections.namedtuple('Hit', ['target_name', 'target_accession', 'bitscore'])

_orf_finder = pyrodigal_gv.ViralGeneFinder(meta=True, mask=True)

def _predict_genes(seq: Tuple[str, str]):
    return (seq[0], len(seq[1]), _orf_finder.find_genes(seq[1]))

class Prodigal:
    def __init__(self, input_file: Path, threads: int, prodigal_output: Path):
        self.input_file = input_file
        self.threads = threads
        self.prodigal_output = prodigal_output
        self.threads = threads
        if not self.prodigal_output.exists():
            self.parallel_write()
    
    def parallel_write(self):
        with Pool(self.threads) as pool, open(self.prodigal_output, "w") as fout, open(self.input_file) as fin:
            input_sequences = SimpleFastaParser(fin)
            for contig, (seq_acc, seq_len, predicted_genes) in enumerate(
                pool.imap(_predict_genes, input_sequences)
            ):
                seq_acc = seq_acc.split()[0]
                for gene_i, gene in enumerate(predicted_genes, 1):
                    header = (
                        f"{seq_acc}|{gene_i} {contig}_{gene_i}_{gene.begin}_{gene.end}_{seq_len}"
                    )
                    print(f">{header}", file=fout)
                    print(gene.translate(include_stop=False), file=fout)

    def proteins(self):
        header_parser = re.compile(r'(.+)\|(\w+) (\d+)_(\d+)_(\d+)_(\d+)_(\d+)')
        with open(self.prodigal_output) as fin:
            for header, protein in SimpleFastaParser(fin):
                head = header_parser.match(header)
                if head is None:
                    sys.exit("Error with header: ", header)
                seq_acc, contig_pos, contig_num, contig_pos, gene_begin, gene_end, contig_len = header_parser.match(header).groups()

                yield Protein(f"{seq_acc}|{contig_pos}", seq_acc, int(contig_num), int(contig_pos), int(gene_begin), int(gene_end), int(contig_len))

def satellite_finder(input_sequence: Path, model: str):
    cmd = ['docker', 'run',
           '-v', f"{input_sequence.parent.absolute()}/:/home/msf",
           #'-u', '501:20',
           '--platform', 'linux/amd64',
           'gempasteur/satellite_finder:0.9.1',
           '--db-type', 'ordered_replicon',
           '--models', model, 
           '--sequence-db', input_sequence.name,
           '-w', '12',
           '--out-dir', model]
    subprocess_return = subprocess.run(cmd, stdout=subprocess.DEVNULL)
    if subprocess_return.returncode != 0:
        print("Error running Satellite Finder")
        sys.exit(subprocess_return.returncode)


def main():

    num_threads = 6
    output_root = Path("data/visualization/validated_data/satellite_finder")
    output_root.mkdir(parents=True, exist_ok=True)
    input_path = Path("data/raw/04_verified_sapis/01_seqs_with_host/individual_seqs/")
    input_sequences = list(input_path.glob("*.fasta"))
    print("Input sequences: ", len(input_sequences))
    with open(output_root / "satellite_finder.tsv", "w") as fout:
        for input_sequence in input_sequences:
            with TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)
                logging.info("Predicting genes with Prodigal")
                prodigal = Prodigal(input_file=input_sequence, threads=num_threads, prodigal_output=tmpdir / "prodigal_output.fasta")
                with open(input_sequence) as fin:
                    for header, seq in SimpleFastaParser(fin):
                        seqid = header.split()[0]
                        break
                hits = 0
                satellite_types = ["PICI", "cfPICI", "P4","PLE"]
                for satellite_type in satellite_types:
                    logging.info(f"Predicting satellites with Satellite Finder for {satellite_type}")
                    satellite_finder(input_sequence=tmpdir / "prodigal_output.fasta", model=satellite_type)
                    with open(tmpdir / satellite_type / 'best_solution_summary.tsv') as fin:
                        for line in fin:
                            if line.startswith("UserReplicon"):
                                hits += int(line.strip().split("\t")[-1])

            logging.info(f"Total hits: {hits} for {seqid}")
            print(input_sequence.name, seqid, hits, sep="\t", file=fout)


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
