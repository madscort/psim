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
# - annotations of each hmm
# - presence/absence matrix of each hmm in each sample

# Clustering steps:
# 1. Take all nucletotide sequences
# 2. Predict genes
# 3. Cluster with mmseqs2

# Original version - only Pfam-A
# 1. Search for hmm profiles for representative in each cluster
# 2. 

# Options:
# 
# Pfam-A, TIGRFAM, KEGG Orthology and COG


# Satellite annotation steps:
# 1. Take all sequences
# 2. Annotate using mmseqs2
# 3. Create presence absence matrix


Protein = namedtuple("Protein", ["genome", "protein", "seq"])
Hit = namedtuple('Hit', ['target_name', 'target_accession', 'evalue'])

def gene_prediction(fasta: Path=None) -> list[Protein]:
    record = SeqIO.read(str(fasta), "fasta")
    proteins = []
    orf_finder = pyrodigal_gv.ViralGeneFinder(meta=True)
    for i, pred in enumerate(orf_finder.find_genes(bytes(record.seq))):
        proteins.append(Protein(genome=record.id, protein=f"{record.id}|{i+1}", seq=pred.translate()))
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
    
    def cluster(self, cluster_mode: int = 1, cov_mode: int = 0, min_cov: float = 0.9, min_seq_id: float = 0.3):
        cmd = ['mmseqs','cluster', '--cluster-mode', f"{cluster_mode}", '--min-seq-id', f"{min_seq_id}", '--cov-mode', f"{cov_mode}",
            '-c', f"{min_cov}", self.out / "db", self.out / "cluster", self.out / "tmp"]
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
    def __init__(self, hmm: Path, db: Path, out: Path):
        self.hmm = hmm
        self.db = db
        self.out = out
        self.out.mkdir(parents=True, exist_ok=True)
    
    def search(self, evalue: float = 1e-3, cpus: int = 6):
        cmd = ['hmmsearch', '--acc', '--tblout', self.out / "hmmsearch_res.txt", '--cpu', f"{cpus}", '-E', f"{evalue}", self.hmm, self.db]
        subprocess_return = subprocess.run(cmd, stdout=subprocess.DEVNULL)
        if subprocess_return.returncode != 0:
            raise Exception(f"Error running hmmsearch: {subprocess_return}")
    
    def parse(self):
        query_hits = {}
        with open(self.out / "hmmsearch_res.txt") as f:
             for line in f:
                if not line.startswith("#"):
                    res = line.split()
                    qname, tname, tacc, evalue = res[0], res[2], res[3], float(res[4])
                    if qname not in query_hits:
                        query_hits[qname] = []
                    query_hits[qname].append(Hit(tname, tacc, evalue))
        return query_hits
    
    def get_best_hits(self):
        best_hits = {}
        all_hits = self.parse()

        for query_name, hits in all_hits.items():
            hits.sort(key=lambda x: x.evalue)
            best_hits[query_name] = hits[0]
        return best_hits
    
    def get_accessions(self):
        best_hits = self.get_best_hits()
        accessions = set([best_hits[hit].target_accession for hit in best_hits])
        return accessions
    
    def get_hmms(self)-> Path:
        accessions = self.get_accessions()
        with open(self.out / "accessions.txt", "w") as f:
            for accession in accessions:
                print(accession, file=f)
        cmd = ['hmmfetch', '-o', self.out / "hmms.hmm", '-f', self.hmm, self.out / "accessions.txt"]
        subprocess_return = subprocess.run(cmd, stdout=subprocess.DEVNULL)
        if subprocess_return.returncode != 0:
            raise Exception(f"Error running hmmfetch: {subprocess_return}")
        
        cmd = ['hmmpress', self.out / "hmms.hmm"]
        subprocess_return = subprocess.run(cmd, stdout=subprocess.DEVNULL)
        if subprocess_return.returncode != 0:
            raise Exception(f"Error running hmmpress: {subprocess_return}")
        return self.out / "hmms.hmm"
    def db_path(self):
        return self.out / "hmms.hmm"

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    project_dir = Path(__file__).resolve().parents[2]
    load_dotenv(find_dotenv())

    dataset = Path("data/processed/10_datasets/dataset_v01")
    sampletable = dataset / "train.tsv"
    output = Path("data/processed/10_datasets/attachings")
    output_path = dataset / "strings"
    samples = get_samples(sampletable)
    output.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output / "proteins.faa", "w") as f_out:
        for sample in samples:
            sample_path = Path(dataset, "train", "satellite_sequences", f"{sample}.fna")
            proteins = gene_prediction(sample_path)
            for protein in proteins:
                print(f">{protein.protein}", file=f_out)
                print(protein.seq, file=f_out)

    mm = MMseqs2(db=output / "proteins.faa", out=output / "mmseqsDB")
    mm.createdb()
    mm.cluster()
    mm.createtsv()
    counts = mm.counts()
    with open(output / "size_dist.tsv", "w") as f_out:
        for protein in counts:
            print(protein, counts[protein], file=f_out, sep="\t")

    # Get representatives sequences for each cluster
    min_cluster_size = 5

    representatives = mm.get_representatives(min_cluster_size)
    logging.info(f"Number of cluster representatives: {len(representatives)}")

    records = get_records_from_identifiers(fasta=output / "proteins.faa", identifiers=representatives)
    SeqIO.write(records, output / "representatives.faa", "fasta")

    # Get hmms for each cluster
    pfam_hmm = Path("data/external/databases/pfam/pfam_A/Pfam-A.hmm")
    hmmer = HMMER(hmm=pfam_hmm, db=output / "representatives.faa", out=output / "hmms")
    hmmer.search()
    original_clusters = set(representatives)
    hits = hmmer.get_best_hits()
    covered_clusters = set([hit for hit in hits])
    accessions = set([hits[hit].target_accession for hit in hits])
    
    print(f"Original clusters: {len(original_clusters)}")
    print(f"Covered clusters: {len(covered_clusters)}")
    print(f"Missing clusters: {len(original_clusters - covered_clusters)}")
    print(f"Number of non-redundant accessions: {len(accessions)}")

    # Get hmms for each accession
    hmmer.get_hmms()
    result_db = hmmer.db_path()

    # Scan all representative sequences with all hmms
    hmmer_scan = HMMER(hmm=result_db, db=output / "representatives.faa", out=output / "hmmer_scan")
    hmmer_scan.search()
    hits = hmmer_scan.get_best_hits()

    hit_clusters = set()
    for hit in hits:
        hit_clusters.add(hit)

    print(f"DB representatives hits: {len(hit_clusters)}")
    print(f"Missing clusters: {len( original_clusters - hit_clusters)}")

    splits = ["train","val","test"]
    data_splits = {
        split: load_split_data(dataset, split) for split in splits
    }

    for split in data_splits:
        print(f"Split: {split}")
        sequences = data_splits[split]['sequences']
        protein_strings = []
        len_strings = []
        with TemporaryDirectory() as tmp_work:
            tmp_work = Path(tmp_work)
            with open(tmp_work / "proteins.faa", "w") as f_out:
                print("Translating sequences...")
                for sequence in tqdm(sequences):
                    protein_string = {}
                    proteins = gene_prediction(sequence)
                    for protein in proteins:
                        protein_string[protein.protein] = "no_hit"
                        print(f">{protein.protein}", file=f_out)
                        print(protein.seq, file=f_out)
                    protein_strings.append(protein_string)
                    len_strings.append(len(protein_string))
            hmmer_scan = HMMER(hmm=result_db, db=tmp_work / "proteins.faa", out=tmp_work / "hmmer_scan")
            hmmer_scan.search()
            hits = hmmer_scan.get_best_hits()
        for contig in protein_strings:
            for pr in contig:
                if pr in hits:
                    contig[pr] = hits[pr].target_accession

        # Get list of lists of accessions:
        accession_string = []
        for contig in protein_strings:
            accession_string.append([contig[pr] for pr in contig])
        for n, seq in enumerate(accession_string):
            assert len(seq) == len_strings[n]
        data_splits[split]['sequences'] = accession_string

    # Save data
    torch.save(data_splits, output_path / "pfam.pt")
