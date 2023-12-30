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

## 2. Profile creation - base version - only Pfam-A
# 1. Search pfam-a for hmm profiles for representative in each cluster
# 2. Create collected database with hmms

## 3. Predictions - base version - hmmsearch/hmmscan:
# 1. Translate dataset and search against hmms
# 2. Create ordered accessions strings for complete dataset
# 3. Pickle and save as .pt


Protein = namedtuple("Protein", ["genome", "protein", "seq"])
Hit = namedtuple('Hit', ['target_name', 'target_accession', 'bitscore'])

def gene_prediction(fasta: Path=None) -> list[Protein]:
    record = SeqIO.read(str(fasta), "fasta")
    proteins = []
    orf_finder = pyrodigal_gv.ViralGeneFinder(meta=True)
    for i, pred in enumerate(orf_finder.find_genes(bytes(record.seq))):
        proteins.append(Protein(genome=record.id, protein=f"{record.id}|{i+1}", seq=pred.translate()))
    return proteins

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
    
    def cluster(self, cluster_mode: int = 0, cov_mode: int = 0, min_cov: float = 0.9, min_seq_id: float = 0.3):
        cmd = ['mmseqs','cluster',
               '--cluster-mode', f"{cluster_mode}",
               '--min-seq-id', f"{min_seq_id}",
               '--cov-mode', f"{cov_mode}",
               '-c', f"{min_cov}",
               '-s', '7',
               '--cluster-steps', '5',
               '--cluster-reassign', '1',
               self.out / "db", self.out / "cluster", self.out / "tmp"]
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
            f.readline()
            counts = Counter([line.strip().split("\t")[0] for line in f])
        return counts
    
    def get_clusters(self, cluster: str):
        with open(self.out / "cluster.tsv") as f:
            f.readline()
            clusters = [line.strip().split("\t")[0] for line in f if line.strip().split("\t")[1] == cluster]
        return clusters
    
    def get_representatives(self, min_size: int):
        counts = self.counts()
        representatives = []
        for cluster in counts:
            if counts[cluster] >= min_size:
                representatives.append(cluster)
        return representatives

class HMMER:
    def __init__(self, hmm: Path, db: Path, out: Path, scan: bool = False):
        self.hmm = hmm
        self.db = db
        self.out = out
        self.out.mkdir(parents=True, exist_ok=True)
        self.wscan = scan
    
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

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    project_dir = Path(__file__).resolve().parents[2]
    load_dotenv(find_dotenv())

    dataset = Path("data/processed/10_datasets/dataset_v02")
    sampletable = dataset / "train.tsv"
    pfam_hmm = Path("data/external/databases/pfam/pfam_A/Pfam-A.hmm")
    output = Path("data/processed/10_datasets/v02/attachings")
    output_path = dataset / "strings" / "pfama"
    samples = get_samples(sampletable)
    output.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)
    
    COLLECT_PROTEINS = False
    CLUSTER = False
    SAVE_REPRESENTATIVES = False
    FIND_HMMS = False
    FETCH_HMMS = False
    SCAN_REPRESENTATIVES = False
    TRANSLATE_DATASET = False
    SCAN_DATASET = True
    ATTACH_TO_DATASET = True

    if COLLECT_PROTEINS:
        logging.info("Collecting proteins...")
        with open(output / "proteins.faa", "w") as f_out:
            for sample in samples:
                sample_path = Path(dataset, "train", "satellite_sequences", f"{sample}.fna")
                proteins = gene_prediction(sample_path)
                for protein in proteins:
                    print(f">{protein.protein}", file=f_out)
                    print(protein.seq, file=f_out)
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

    # Get representatives sequences for each cluster
    min_cluster_size = 5

    representatives = mm.get_representatives(min_cluster_size)
    logging.info(f"Number of cluster representatives: {len(representatives)}")

    if SAVE_REPRESENTATIVES:
        logging.info("Saving representatives...")
        records = get_records_from_identifiers(fasta=output / "proteins.faa", identifiers=representatives)
        SeqIO.write(records, output / "representatives.faa", "fasta")
        logging.info("Done saving representatives.")

    # Get hmms for each cluster

    hmmer = HMMER(hmm=pfam_hmm, db=output / "representatives.faa", out=output / "hmms")
    if FIND_HMMS:
        logging.info("Finding hmms...")
        hmmer.search()
        logging.info("Done finding hmms.")

    original_clusters = set(representatives)
    hits = hmmer.get_best_hits()
    covered_clusters = set([hit for hit in hits])
    accessions = set([hits[hit].target_accession for hit in hits])
    
    print(f"Original clusters: {len(original_clusters)}")
    print(f"Covered clusters: {len(covered_clusters)}")
    print(f"Missing clusters: {len(original_clusters - covered_clusters)}")
    print(f"Number of non-redundant accessions: {len(accessions)}")

    # Get hmms for each accession
    if FETCH_HMMS:
        logging.info("Fetching hmms...")
        hmmer.get_hmms()
        logging.info("Done fetching hmms.")
    result_db = hmmer.db_path()

    # Scan all representative sequences with all hmms
    hmmer_scan = HMMER(hmm=result_db, db=output / "representatives.faa", out=output / "hmmer_scan")
    
    if SCAN_REPRESENTATIVES:
        logging.info("Scanning representatives...")
        hmmer_scan.search()
        logging.info("Done scanning representatives.")
    
    hits = hmmer_scan.get_best_hits()

    hit_clusters = set([hit for hit in hits])

    print(f"DB representatives hits: {len(hit_clusters)}")
    print(f"Missing clusters: {len( original_clusters - hit_clusters)}")

    accessions = set([hits[hit].target_accession for hit in hits])
    print(f"Number of non-redundant accessions used: {len(accessions)}")
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
                        protein_string[protein] = hits[protein].target_accession
                hmm_data_splits[split]['sequences'][n] = list(protein_string.values())

        # Get vocabulary from all non-redundant accessions + 'no_hit'
        vocab = set()  
        for split in splits:
            vocab.update(x for seq in hmm_data_splits[split]['sequences'] for x in seq)
        vocab_map = {name: i for i, name in enumerate(vocab)}
        vocab.remove("no_hit")
        print(f"Number of non-redundant accessions in HMM database: {len(accessions)}")
        print(f"Number of non-redundant accessions used in dataset: {len(vocab)}")
        print(f"Accessions numbers not used: {len(accessions - vocab)}")
        torch.save(hmm_data_splits, output_path / "dataset.pt")
        torch.save(vocab_map, output_path / "vocab_map.pt")

        logging.info("Done attaching dataset.")
    

def add_missing_clusters():
    # Find hmms for missing clusters:
    missing_representatives = original_clusters - hit_clusters

    # Test with checkV hmm database
    checkV = Path("data/external/databases/checkv/checkv-db-v1.5/hmm_db/checkv_hmms")
    with TemporaryDirectory() as tmp_work:
        tmp_work = Path(tmp_work)
        records = get_records_from_identifiers(fasta=output / "proteins.faa", identifiers=missing_representatives)
        SeqIO.write(records, tmp_work / "representatives.faa", "fasta")
        hits_list = []
        for profile_set in range(1,81):
            print(f"Profile set: {profile_set}")
            hmmer_scan = HMMER(hmm=checkV / f"{profile_set}.hmm", db=tmp_work / "representatives.faa", out=tmp_work / "hmmer_scan")
            hmmer_scan.search()
            hits = hmmer_scan.get_best_hits()
            print(hits)
            if profile_set == 10:
                break
    # Get best hits for each representative
    for hits in hits_list:
        for hit in hits:
            hit_clusters.add(hit)
    print(len(hits_list))