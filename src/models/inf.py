from pathlib import Path
from pytorch_lightning import seed_everything
from tqdm import tqdm
from multiprocessing import Pool
from typing import List, Tuple
from tempfile import TemporaryDirectory
import torch
import logging
import sys
import pandas as pd
from torch.nn.functional import one_hot
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

translate = str.maketrans("ACGTURYKMSWBDHVN", "0123444444444444")
Protein = collections.namedtuple("Protein", ["accession", "contig", "contig_num", "contig_order", "start", "end", "contig_len"])
Sequence = collections.namedtuple("Sequence", ["id", "host", "seq"])
Hit = collections.namedtuple('Hit', ['target_name', 'target_accession', 'bitscore'])

_orf_finder = pyrodigal_gv.ViralGeneFinder(meta=True, mask=True)

def _predict_genes(seq: Tuple[str, str]):
    return (seq[0], len(seq[1]), _orf_finder.find_genes(seq[1]))

prodigal_root = Path("data/visualization/validated_data/prodigal/")
prodigal_root.mkdir(parents=True, exist_ok=True)
prodigal_output = prodigal_root / "proteins.faa"
class Prodigal:
    def __init__(self, input_file: Path, threads: int):
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

def main():

    seed_everything(1)
    device = torch.device("cpu")

    output_root = Path("data/visualization/real_data/transformer")
    output_root.mkdir(parents=True, exist_ok=True)

    model_database = Path("data/processed/10_datasets/attachings_blitsDB/mmseqsprofileclusterDB/profileClusterDB")
    model_database_out = output_root / "db_out"
    model_database_out.mkdir(parents=True, exist_ok=True)

    database_type = "mmseqs"

    chunk_size = 25000
    transformer = True
    num_threads = 6
    protein_stride = 1
    satellite_size_filter = 1000
    satellite_protein_size_filter = 10

    # Takes a fasta file and returns predictions per sequence.
    input_test = Path("data/visualization/real_data/sludge_25kb_10000.fna")
    logging.info("Predicting genes with Prodigal")
    prodigal = Prodigal(input_file=input_test, threads=num_threads)
    logging.info("Predicting genes with Prodigal done")
    proteins = prodigal.proteins()
    
    model = SequenceModule.load_from_checkpoint(checkpoint_path=Path("models/transformer/blitsDB_a4q0ml04.ckpt").absolute(),
                                                map_location=device)
    model.to(device)
    model.eval()

    if transformer:
        vocab_map = model.vocab_map
        max_seq_length = 56 # LATER ADD TO: model.max_seq_length --> Right now the model determines max seq length of predictions..

    if database_type == "hmm":
        mDB = HMMER(hmm=model_database, db=output_root / "proteins.faa", out=model_database_out, scan=True)
        mDB.scan()
    elif database_type == "mmseqs":
        mDB = SMMseqs2(input_fasta=prodigal_output, mmseqs_db=model_database, output_dir=model_database_out)
        if not Path(model_database_out / "result.m8").exists():
            logging.info("Running mmseqs2 search")
            mDB.run_alignment(threads=num_threads, evalue=1e-3) 
            logging.info("Search done")

    hits = mDB.get_best_hits()
    contigs = {x.split("|")[0] for x in hits.keys()} # Convert to class function

    logging.info("Number of contigs with hits: %s", len(contigs))

    current_contig = None
    logging.info("Starting prediction")
    for protein in proteins:

        if protein.contig != current_contig:
            if current_contig is None:
                protein_sequence = []
                current_contig = protein.contig
            elif len(protein_sequence) > 0:
                found = False
                super_protein = False
                contig_len = protein_sequence[0].contig_len
                predictions_sequence = [0] * contig_len
                predictions_proteins = [0] * len(protein_sequence)
                for i in range(0, len(protein_sequence)-protein_stride+1, protein_stride):
                    if (contig_len - protein_sequence[i].start) < chunk_size or (len(protein_sequence) - i) < satellite_protein_size_filter:
                        break
                    # Eat proteins from index until chunk_size is reached:
                    eating_proteins = True
                    current_index = i
                    length_dna = 0
                    model_string = []
                    protein_string = []
                    # Proteins can apparently be +25kb - this results in a protein eating-stop ie. length gets shorter.
                    # up until the long protein. It will finally die, because it cannot jump past it.
                    
                    while eating_proteins and current_index < len(protein_sequence) and len(model_string) < max_seq_length:
                        this_protein = protein_sequence[current_index]
                        if this_protein.end - this_protein.start > chunk_size:
                            super_protein = True
                        if length_dna + this_protein.end - this_protein.start < chunk_size:
                            length_dna += this_protein.end - this_protein.start
                            if this_protein.accession in hits:
                                if database_type == "hmm":
                                    model_string.append(hits[this_protein.accession].target_name)
                                elif database_type == "mmseqs":
                                    target_name = hits[this_protein.accession].target_name.replace("|", "_")
                                    model_string.append(target_name)
                            else:
                                model_string.append('no_hit')
                            protein_string.append((this_protein.start, this_protein.end))
                            current_index += 1
                        else:
                            eating_proteins = False
                    if super_protein and len(model_string) < satellite_protein_size_filter:
                        continue
                    if transformer:
                        encoded_string = torch.tensor(encode_sequence(model_string, vocab_map))
                        sequence = {"seqs": encoded_string.unsqueeze(0)}

                    prediction = model(sequence)
                    if prediction.argmax(dim=1).tolist()[0] == 1:
                        found = True
                        for x in range(protein_string[0][0], protein_string[-1][1]):
                            predictions_sequence[x] += 1
                        for x in range(len(protein_string)):
                            predictions_proteins[i+x] += 1
                if found:
                    # Use prediction_proteins list for visualisation file, print each value to file
                    with open(output_root / "visualisation.tsv", "a") as fout:
                        for prot_i, prot in enumerate(protein_sequence):
                            print(prot_i,
                            prot.contig,
                            predictions_proteins[prot_i],
                            sep="\t",
                            file=fout)
                    # Use prediction_sequence list for coordinate determination.
                    with open(output_root / "coordinates.tsv", "a") as fout:
                        satellite = False
                        for pred_i, pred in enumerate(predictions_sequence):
                            if pred > 0 and not satellite:
                                satellite = True
                                start_coord = pred_i
                            elif pred == 0 and satellite:
                                satellite = False
                                end_coord = pred_i
                                if end_coord - start_coord > satellite_size_filter:
                                    print(current_contig,
                                    start_coord,
                                    end_coord,
                                    sep="\t",
                                    file=fout)
            
                current_contig = protein.contig
                protein_sequence = []
                if protein.contig not in contigs:
                    continue
                protein_sequence.append(protein)
            else:
                current_contig = protein.contig
                protein_sequence = []
                if protein.contig not in contigs:
                    continue
                protein_sequence.append(protein)
        elif protein.contig not in contigs:
            continue
        elif protein.contig == current_contig:
            protein_sequence.append(protein)

if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
