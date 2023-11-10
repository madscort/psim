import logging
from Bio import SeqIO
import pyrodigal_gv
import sys
import pandas as pd
import torch
import subprocess
from pathlib import Path
from tqdm import tqdm
from dotenv import find_dotenv, load_dotenv
from collections import namedtuple, Counter
from tempfile import TemporaryDirectory
from datetime import datetime
import multiprocessing
from typing import List, Tuple

# 2023-11-01 mads

# This script is used to build a pangenome and a corresponding set of HMMs
# to predict the presence of genes in other sequences.
# Takes
# - sampletable with sample IDs
# - folder with individual nucleotide sequences
# Outputs
# - set of hmms in single files
# - pickled dataset file with strings of hmm accession matches

## 1. Clustering steps:
# 1. Take all nucletotide sequences
# 2. Predict genes
# 3. Cluster with mmseqs2

## 2. Profile creation - de-novo version:
# 1. Collect proteins for each cluster in single fasta file -> fasta format
# 2. Align clusters with clustal omega -> clustal format
# 3. Build hmm profiles with hmmmake -> hhm format
# 4. Convert to hhm with hhmake -> hmm format

## 2+. Profile creation - de-novo + enrichment version:
# 1. Collect proteins for each cluster in single fasta file -> fasta format
# 2. Align clusters with clustal omega -> clustal format
# 3. Build hmm profiles with hhmake -> hhm format
# 4. Convert to hmm with hhmake -> hmm format


## 3. Predictions - base version - hmmsearch/hmmscan:
# 1. Translate dataset -> fasta format
# 2. Search against hmms -> hmmsearch
# 2. Create ordered pfama accessions strings for complete dataset
# 3. Pickle and save as .pt

Protein = namedtuple("Protein", ["genome", "protein", "seq"])
Hit = namedtuple('Hit', ['target_name', 'target_accession', 'bitscore'])

def gene_prediction(sample_path: Path) -> List[Protein]:
    record = SeqIO.read(str(sample_path), "fasta")
    proteins = []
    orf_finder = pyrodigal_gv.ViralGeneFinder(meta=True)
    for i, pred in enumerate(orf_finder.find_genes(str(record.seq))):
        proteins.append(Protein(genome=record.id, protein=f"{record.id}|{i+1}", seq=pred.translate(include_stop=False)))
    return proteins

def write_proteins_to_file(proteins: List[Protein], f_out):
    for protein in proteins:
        print(f">{protein.protein}", file=f_out)
        print(protein.seq, file=f_out)

def parallel_gene_prediction(dataset: Path, output: Path, samples: List[str], threads: int) -> None:

    sample_paths = [Path(dataset, "train", "satellite_sequences", f"{sample}.fna") for sample in samples]

    with multiprocessing.Pool(threads) as pool:
        results = pool.map(gene_prediction, sample_paths)

    with open(output / "proteins.faa", "w") as f_out:
        for proteins in results:
            write_proteins_to_file(proteins, f_out)


def gene_prediction_string(sequence: str=None) -> list[Protein]:
    proteins = []
    orf_finder = pyrodigal_gv.ViralGeneFinder(meta=True)
    for i, pred in enumerate(orf_finder.find_genes(sequence=sequence)):
        proteins.append(Protein(genome="string", protein=f"string|{i+1}", seq=pred.translate()))
    return proteins

def get_samples(sampletable: Path=None) -> list[Path]:
    samples = []
    with open(sampletable) as f:
        f.readline()
        for line in f:
            sample = line.strip().split("\t")
            if sample[-1] == "1":
                samples.append(sample[0])
    return samples

def get_records_from_identifiers(fasta: Path = None, identifiers: list = None):
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

def load_split_data(dataset, split):
    df = pd.read_csv(dataset / f"{split}.tsv", sep="\t", header=0, names=["id", "type", "label"])
    sequences = [dataset / split / "sequences" / f"{id}.fna" for id in df['id'].values]
    labels = torch.tensor(df['label'].values)
    return {'sequences': sequences, 'labels': labels}

class MMseqs2:
    def __init__(self, db: Path, out: Path):
        self.db = db
        self.out = out
        self.out.mkdir(parents=True, exist_ok=True)
    
    def createdb(self):
        cmd = ["mmseqs", "createdb", self.db, self.out / "db"]
        subprocess_return = subprocess.run(cmd, stdout=subprocess.DEVNULL)
        if subprocess_return.returncode != 0:
            raise Exception(f"Error running mmseqs2: {subprocess_return}")
    
    def cluster(self, cluster_mode: int = 0, min_seq_id: float = 0.5, cov_mode: int = 0, min_cov: float = 0.8):
        cmd = ['mmseqs','cluster', '--cluster-mode', f"{cluster_mode}", '--min-seq-id', f"{min_seq_id}", '--cov-mode', f"{cov_mode}",
            '-c', f"{min_cov}", '-s', '5.5', '-e', '1e-5', '--cluster-reassign', '1', self.out / "db", self.out / "cluster", self.out / "tmp"]
        subprocess_return = subprocess.run(cmd)
        if subprocess_return.returncode != 0:
            raise Exception(f"Error running mmseqs2: {subprocess_return}")
    
    def createtsv(self):
        cmd = ['mmseqs','createtsv', self.out / "db", self.out / "db", self.out / "cluster", self.out / "cluster.tsv"]
        subprocess_return = subprocess.run(cmd, stdout=subprocess.DEVNULL)
        if subprocess_return.returncode != 0:
            raise Exception(f"Error running mmseqs2: {subprocess_return}")
    
    def counts(self):
        with open(self.out / "cluster.tsv") as f:
            counts = Counter([line.strip().split("\t")[0] for line in f])
        return counts
    
    def get_clusters(self, representative: str):
        with open(self.out / "cluster.tsv") as f:
            cluster_proteinIDs = [line.strip().split("\t")[1] for line in f if line.strip().split("\t")[0] == representative]
        return cluster_proteinIDs
    
    def get_representatives(self, min_size: int):
        counts = self.counts()
        representatives = []
        for cluster in counts:
            if counts[cluster] >= min_size:
                representatives.append(cluster)
        return representatives

class HMMER:
    def __init__(self, hmm: Path, db: Path, out: Path, scan: bool = False, by_name: bool = False):
        self.hmm = hmm
        self.db = db
        self.out = out
        self.wscan = scan
        self.by_name = by_name
    
    def search(self, evalue: float = 1e-3, cpus: int = 6):
        cmd = ['hmmsearch', '--acc', '--tblout', self.out / "hmmsearch_res.txt", '--cpu', f"{cpus}", '-E', f"{evalue}", self.hmm, self.db]
        subprocess_return = subprocess.run(cmd, stdout=subprocess.DEVNULL)
        if subprocess_return.returncode != 0:
            raise Exception(f"Error running hmmsearch: {subprocess_return}")
    
    def scan(self, evalue: float = 1e-3, cpus: int = 6):
        cmd = ['hmmscan', '--acc', '--tblout', self.out / "hmmsearch_res.txt", '--cpu', f"{cpus}", '-E', f"{evalue}", self.hmm, self.db]
        subprocess_return = subprocess.run(cmd, stdout=subprocess.DEVNULL)
        if subprocess_return.returncode != 0:
            raise Exception(f"Error running hmmscan: {subprocess_return}")
        
    def build(self):
        cmd = ['hmmbuild', self.out, self.db]
        subprocess_return = subprocess.run(cmd, stdout=subprocess.DEVNULL)
        if subprocess_return.returncode != 0:
            raise Exception(f"Error running hmmbuild: {subprocess_return}")
    
    def concat_hmms(self, input_path: Path, output_path: Path) -> Path:
        # Concat all hmms into one file:
        hmms = list(input_path.glob("*.hmm"))
        with open(output_path, "w") as f_out:
            for hmm in hmms:
                with open(hmm) as f_in:
                    for line in f_in:
                        print(line.strip(), file=f_out)
                print("\n", file=f_out)
        
        cmd = ['hmmpress', output_path]
        subprocess_return = subprocess.run(cmd, stdout=subprocess.DEVNULL)
        if subprocess_return.returncode != 0:
            raise Exception(f"Error running hmmpress: {subprocess_return}")
        return output_path

    def parse(self):
        query_hits = {}
        with open(self.out / "hmmsearch_res.txt") as f:
             for line in f:
                if not line.startswith("#"):
                    res = line.split()
                    if self.wscan:
                        qname, tname, tacc, bitscore = res[2], res[0], res[1], float(res[5])
                    else:
                        qname, tname, tacc, bitscore = res[0], res[2], res[3], float(res[5])
                    if qname not in query_hits:
                        query_hits[qname] = []
                    query_hits[qname].append(Hit(tname, tacc, bitscore))
        return query_hits
    
    def get_best_hits(self):
        best_hits = {}
        all_hits = self.parse()

        for query_name, hits in all_hits.items():
            hits.sort(key=lambda x: x.bitscore, reverse=True)
            best_hits[query_name] = hits[0]
        return best_hits
    
    def get_accessions(self):
        best_hits = self.get_best_hits()
        accessions = set([best_hits[hit].target_accession for hit in best_hits])
        return accessions
    
    def get_hmms(self, accessions: set = None, output_path: Path = None) -> Path:
        if accessions is None:
            accessions = self.get_accessions()
            output_path = self.out

        output_path.mkdir(parents=True, exist_ok=True)
        
        with open(output_path / "accessions.txt", "w") as f:
            for accession in accessions:
                print(accession, file=f)
        cmd = ['hmmfetch', '-o', output_path / "hmms.hmm", '-f', self.hmm, output_path / "accessions.txt"]
        subprocess_return = subprocess.run(cmd, stdout=subprocess.DEVNULL)
        if subprocess_return.returncode != 0:
            raise Exception(f"Error running hmmfetch: {subprocess_return}")
        
        cmd = ['hmmpress', output_path / "hmms.hmm"]
        subprocess_return = subprocess.run(cmd, stdout=subprocess.DEVNULL)
        if subprocess_return.returncode != 0:
            raise Exception(f"Error running hmmpress: {subprocess_return}")
        return output_path / "hmms.hmm"
    def db_path(self):
        return self.out / "hmms.hmm"

class CLUSTALO:
    def __init__(self, fasta: Path, out: Path, threads: int = 1):
        self.fasta = fasta
        self.out = out
        self.threads = threads
    def align(self):
        cmd = ['clustalo', '-i', self.fasta, '-o', self.out, '--outfmt=st', f"--threads={self.threads}", '--force']
        subprocess_return = subprocess.run(cmd)
        if subprocess_return.returncode != 0:
            raise Exception(f"Error running clustalo: {subprocess_return}")

class HHSUITE:
    def __init__(self, input: Path, out: Path):
        self.input = input
        self.out = out
    def convert(self, in_format, out_format):
        cmd = ['reformat.pl', in_format, out_format, '-i', self.input, '-o', self.out]
        subprocess_return = subprocess.run(cmd, stdout=subprocess.DEVNULL)
        if subprocess_return.returncode != 0:
            raise Exception(f"Error running hhconvert: {subprocess_return}")
    def make(self):
        cmd = ['hhmake', '-i', self.input, '-o', self.out]
        subprocess_return = subprocess.run(cmd, stdout=subprocess.DEVNULL)
        if subprocess_return.returncode != 0:
            raise Exception(f"Error running hhmake: {subprocess_return}")

def clean_prodigal(input):
    cmd = ['sed','-i','','s/*//g', input]
    subprocess_return = subprocess.run(cmd)
    if subprocess_return.returncode != 0:
        raise Exception(f"Error running sed: {subprocess_return}")

def get_one_string(sequence, hmm_db):
    with TemporaryDirectory() as tmp_work:
        tmp_work = Path(tmp_work)
        with open(tmp_work / "proteins.faa", "w") as f_out:
            protein_string = {}
            proteins = gene_prediction_string(sequence)
            for protein in proteins:
                protein_string[protein.protein] = "no_hit"
                print(f">{protein.protein}", file=f_out)
                print(protein.seq, file=f_out)
        hmmer_scan = HMMER(hmm=hmm_db, db=tmp_work / "proteins.faa", out=tmp_work / "hmmer_scan", scan=True)
        hmmer_scan.scan()
        hits = hmmer_scan.get_best_hits()
        for pr in protein_string:
            if pr in hits:
                protein_string[pr] = hits[pr].target_accession
        return list(protein_string.values())

def get_hmms(accessions: set = None, hmm_db: Path = None, output_path: Path = None) -> Path:

        output_path.mkdir(parents=True, exist_ok=True)
        
        with open(output_path / "accessions.txt", "w") as f:
            for accession in accessions:
                print(accession, file=f)
        cmd = ['hmmfetch', '-o', output_path / "hmms.hmm", '-f', hmm_db, output_path / "accessions.txt"]
        subprocess_return = subprocess.run(cmd, stdout=subprocess.DEVNULL)
        if subprocess_return.returncode != 0:
            raise Exception(f"Error running hmmfetch: {subprocess_return}")
        
        cmd = ['hmmpress', output_path / "hmms.hmm"]
        subprocess_return = subprocess.run(cmd, stdout=subprocess.DEVNULL)
        if subprocess_return.returncode != 0:
            raise Exception(f"Error running hmmpress: {subprocess_return}")
        return output_path / "hmms.hmm"

class SMMseqs2:
    def __init__(self, input_fasta: Path, mmseqs_db: Path, output_dir: Path) -> None:
        self.input_fasta = input_fasta
        self.mmseqs_db = mmseqs_db
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

    def run_alignment(self, threads: int, evalue: float) -> None:
        # Define paths for the created database and alignment results
        query_db = self.output_dir / "query_db"
        align_db = self.output_dir / "align_db"
        besthit_db = self.output_dir / "besthit_db"
        result_file = self.output_dir / "result.m8"

        # Create a database from the input FASTA file
        createdb_command = ["mmseqs", "createdb", self.input_fasta, query_db]
        self._run_command(createdb_command, "Failed to create query database")

        # Run cluster workflow (includes prefilter)
        align_command = [
            "mmseqs", "search",
            query_db,
            self.mmseqs_db,
            align_db,
            self.output_dir,
            "--threads", str(threads),
            "-e", str(evalue),
            "-s", "6.4",
            "-c", "0.2",
            "--cov-mode", "1"
        ]
        self._run_command(align_command, "Alignment failed")

        # Filter results to get best hit:
        besthit_command = [
            "mmseqs", "filterdb",
            align_db,
            besthit_db,
            "--extract-lines", "1"
        ]
        self._run_command(besthit_command, "Best hit filtering failed")

        # Convert the binary results to a human-readable format
        convertalis_command = [
            "mmseqs", "convertalis",
            query_db,
            self.mmseqs_db,
            besthit_db,
            result_file,
            "--format-output", "query,target,evalue,bits"
        ]
        self._run_command(convertalis_command, "Result conversion failed")

        print(f"Alignment and result conversion completed. Results are in: {result_file}")

    def _run_command(self, command: list, error_message: str) -> None:
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            raise Exception(f"{error_message}: {e}")

def trimal(input: Path, output: Path):
    # Trim alignments
    cmd = ['trimal', '-in', input, '-out', output, '-automated1']
    subprocess_return = subprocess.run(cmd, stdout=subprocess.DEVNULL)
    if subprocess_return.returncode != 0:
        raise Exception(f"Error running trimal: {subprocess_return}")

def index_hmms(input: List, output: Path):
    # Index using ffindex_build
    cmd = ['ffindex_build', output / "hmm", output / "hmm.index"] + input
    subprocess_return = subprocess.run(cmd, stdout=subprocess.DEVNULL)
    if subprocess_return.returncode != 0:
        raise Exception(f"Error running ffindex_build: {subprocess_return}")

def convertprofile(input: Path, output: Path):
    # Convert to hmmer format using hmmer2h3m
    cmd = ['mmseqs', 'convertprofiledb', input, output]
    subprocess_return = subprocess.run(cmd, stdout=subprocess.DEVNULL)
    if subprocess_return.returncode != 0:
        raise Exception(f"Error running hmmer2h3m: {subprocess_return}")

def indexprofileDB(input: Path, out: Path):
    # Index mmseqs profileDB
    cmd = ['mmseqs', 'createindex', input, out]
    subprocess_return = subprocess.run(cmd, stdout=subprocess.DEVNULL)
    if subprocess_return.returncode != 0:
        raise Exception(f"Error running mmseqs createindex: {subprocess_return}")

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    project_dir = Path(__file__).resolve().parents[2]
    load_dotenv(find_dotenv())

    dataset = Path("data/processed/10_datasets/dataset_v01")
    sampletable = dataset / "train.tsv"
    pfam_hmm = Path("data/external/databases/pfam/pfam_A/Pfam-A.hmm")
    output = Path("data/processed/10_datasets/attachings_mmDB")
    output_path = dataset / "strings" / "mmDB"
    samples = get_samples(sampletable)
    output.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)
    min_cluster_size = 5
    result_db = output_path / "hmms" / "hmms.hmm"
    result_db.parent.mkdir(parents=True, exist_ok=True)
    num_threads = 4
    
    ## 1: Clustering
    COLLECT_PROTEINS = False
    CLUSTER = False
    SAVE_REPRESENTATIVES = False
    COLLECT_CLUSTERS = False
    ALIGN_CLUSTERS = False
    TRIM_ALIGNMENTS = False
    CONVERT_ALIGNMENTS = False
    MAKE_HMMS = False
    INDEX_HMMS = False
    CONVERT_HMMS = False

    MMSEARCH_REPRESENTATIVES = True

    TRANSLATE_DATASET = False
    SCAN_DATASET = False
    ATTACH_TO_DATASET = False

    if COLLECT_PROTEINS:
        logging.info("Collecting proteins...")
        parallel_gene_prediction(dataset, output, samples, num_threads)
        logging.info("Done collecting proteins.")

    mm = MMseqs2(db=output / "proteins.faa", out=output / "mmseqsDB")
    if CLUSTER:
        logging.info("Clustering...")
        mm.createdb()
        mm.cluster()
        mm.createtsv()
        counts = mm.counts()
        with open(output / "size_dist.tsv", "w") as f_out:
            for protein in counts:
                print(protein, counts[protein], file=f_out, sep="\t")
        logging.info("Done clustering.")
    
    representatives = mm.get_representatives(min_cluster_size)
    logging.info(f"Number of clusters: {len(representatives)}")

    if SAVE_REPRESENTATIVES:
        logging.info("Saving representatives...")
        records = get_records_from_identifiers(fasta=output / "proteins.faa", identifiers=representatives)
        SeqIO.write(records, output / "representatives.faa", "fasta")
        logging.info("Done saving representatives.")

    if COLLECT_CLUSTERS:
        logging.info("Collecting clusters...")
        
        for representative in tqdm(representatives):
            cluster = mm.get_clusters(representative)
            cluster_path = output / "clusters" / f"{representative.replace('|','_')}.faa"
            cluster_path.parent.mkdir(parents=True, exist_ok=True)
            records = get_records_from_identifiers(fasta=output / "proteins.faa", identifiers=cluster)
            SeqIO.write(records, cluster_path, "fasta")
            clean_prodigal(cluster_path)
        logging.info("Done collecting clusters.")

    if ALIGN_CLUSTERS:
        logging.info("Aligning clusters...")
        alignment_root = output / "alignments"
        alignment_root.mkdir(parents=True, exist_ok=True)
        for representative in tqdm(representatives):
            fasta_input = output / "clusters" / f"{representative.replace('|','_')}.faa"
            alignment_output = alignment_root / f"{representative.replace('|','_')}.st"
            clustalo = CLUSTALO(fasta=fasta_input, out=alignment_output)
            clustalo.align()

    # then cat stockholm files
    # then mmseqs convertmsa
    # then mmseqs msa2profile
    # then mmseqs createindex --match-mode 1

    if TRIM_ALIGNMENTS:
        logging.info("Trimming alignments...")
        trim_root = output / "trim_alignments"
        trim_root.mkdir(parents=True, exist_ok=True)
        for representative in tqdm(representatives):
            alignment_input = output / "alignments" / f"{representative.replace('|','_')}.clu"
            trim_output = trim_root / f"{representative.replace('|','_')}.clu"
            trimal(alignment_input, trim_output)
        logging.info("Done trimming alignments.")

    if CONVERT_ALIGNMENTS:
        logging.info("Converting alignments to A3M...")
        a3m_root = output / "alignments_a3m"
        a3m_root.mkdir(parents=True, exist_ok=True)
        for representative in tqdm(representatives):
            clu_input = output / "trim_alignments" / f"{representative.replace('|','_')}.clu"
            a3m_output = a3m_root / f"{representative.replace('|','_')}.a3m"
            hh = HHSUITE(input=clu_input, out=a3m_output)
            hh.convert(in_format="clu", out_format="a3m")

    if MAKE_HMMS:
        logging.info("Making hmms...")
        hmm_root = output / "hmms"
        hmm_root.mkdir(parents=True, exist_ok=True)
        for representative in tqdm(representatives):
            msa_input = output / "alignments_a3m" / f"{representative.replace('|','_')}.a3m"
            hmm_output = hmm_root / f"{representative.replace('|','_')}.hhm"
            hh = HHSUITE(input=msa_input, out=hmm_output)
            hh.make()

    if INDEX_HMMS:
        logging.info("Indexing hmms...")
        index_root = output / "hmm_index"
        index_root.mkdir(parents=True, exist_ok=True)
        hmm_input = [output / "hmms" / f"{representative.replace('|','_')}.hhm" for representative in representatives]
        index_hmms(hmm_input, index_root)
        logging.info("Done indexing hmms.")

    if CONVERT_HMMS:
        logging.info("Converting hmms...")
        hmm_root = output / "mmseqsprofileDB"
        hmm_root.mkdir(parents=True, exist_ok=True)
        indexed_input = output / "hmm_index" / "hmm"
        profileDB = output / "mmseqsprofileDB" / "profileDB"
        convertprofile(indexed_input, profileDB)
        logging.info("Done converting hmms.")


    if MMSEARCH_REPRESENTATIVES:
        logging.info("Searching representatives...")
        mm = SMMseqs2(input_fasta=output / "all_proteins.faa", mmseqs_db=output / "mmseqsprofileDB" / "profileDB", output_dir=output / "mmsearch")
        mm.run_alignment(threads=num_threads, evalue=1e-3)
        logging.info("Done searching representatives.")
    sys.exit()

    # Scan all representative sequences with all hmms
    hmmer_scan_out = output / "hmmer_scan"
    hmmer_scan_out.mkdir(parents=True, exist_ok=True)
    hmmer_scan = HMMER(hmm=result_db, db=output / "representatives.faa", out=output / "hmmer_scan")
    
    splits = ["train", "val", "test"]

    if TRANSLATE_DATASET:
        logging.info("Translating dataset...")
        data_splits = {
            split: load_split_data(dataset, split) for split in splits
        }

        translated_data_splits = {
            split: {'sequences': [None]*len(data_splits[split]['labels']), 'labels': data_splits[split]['labels']} for split in splits
        }

        for split in splits:
            print(f"Split: {split}")
            
            sequences = data_splits[split]['sequences']
            with open(output / f"{split}_split.faa", "w") as f_out:
                print("Translating sequences...")
                
                for n, sequence in enumerate(tqdm(sequences)):
                    
                    protein_string = {}
                    proteins = gene_prediction(sequence)
                    
                    for protein in proteins:
                        protein_string[protein.protein] = "no_hit"
                        print(f">{protein.protein}", file=f_out)
                        print(protein.seq, file=f_out)
                    
                    translated_data_splits[split]['sequences'][n] = protein_string
        torch.save(translated_data_splits, output / "translated.pt")
        logging.info("Done translating dataset.")

    for split in splits:
        out_path = output / f"hmmer_{split}_split"
        out_path.mkdir(parents=True, exist_ok=True)

    hmm_splits = { split: HMMER(hmm=result_db, db=output / f"{split}_split.faa", out=output / f"hmmer_{split}_split", scan=True) for split in splits}
    
    if SCAN_DATASET:
        logging.info("Scanning dataset...")
        for split in splits:
            print(f"Split: {split}")
            hmm_splits[split].scan()
        logging.info("Done scanning dataset.")

    if ATTACH_TO_DATASET:
        logging.info("Attaching dataset...")
        hmm_data_splits = torch.load(output / "translated.pt")

        for split in splits:
            print(f"Split: {split}")
            hits = hmm_splits[split].get_best_hits()
            for n, protein_string in enumerate(tqdm(hmm_data_splits[split]['sequences'])):
                for protein in protein_string:
                    if protein in hits:
                        protein_string[protein] = hits[protein].target_name
                hmm_data_splits[split]['sequences'][n] = list(protein_string.values())

        # Get vocabulary from all non-redundant accessions + 'no_hit'
        vocab = set()  
        for split in splits:
            vocab.update(x for seq in hmm_data_splits[split]['sequences'] for x in seq)
        vocab_map = {name: i for i, name in enumerate(vocab)}
        vocab.remove("no_hit")
        accessions_renamed = set([i.replace('|','_') for i in accessions])
        print(f"Number of non-redundant accessions in HMM database: {len(accessions_renamed)}")
        print(f"Number of non-redundant accessions used in dataset: {len(vocab)}")
        print(f"Accessions numbers not used: {len(accessions_renamed - vocab)}")
        print(f"Accessions numbers not used: {accessions_renamed - vocab}")
        torch.save(hmm_data_splits, output_path / "dataset.pt")
        torch.save(vocab_map, output_path / "vocab_map.pt")

        logging.info("Done attaching dataset.")
