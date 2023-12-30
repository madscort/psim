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

## 2+. Profile creation - de-novo + enrichment version:
# 1. Collect proteins for each cluster in single fasta file -> fasta format
# 2. Align clusters with clustal omega -> clustal format -> reformat.pl -> a3m format
# 3. Run HHblits on each alignment -> a3m format
# 4. Reformat to stockholm and concatenate all alignments -> reformat.pl -> stockholm format -> concat all alignments
# 5. Create profileDB

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
    
    def prep_pcDB(self, outf: Path, profileDB: Path):
        self.outPC = outf
        self.outPC.mkdir(parents=True, exist_ok=True)
        self.profileDB = profileDB
        self.representativesDB = self.outPC / "representativesDB"
        self.profilesearchDB = self.outPC / "profileSearchDB"
        self.profileCluster = self.outPC / "profileCluster"
        self.profileClusterDB = self.outPC / "profileClusterDB"

    def create_repDB(self):
        cmd = ['mmseqs', 'profile2consensus', self.profileDB, self.outPC / "representativesDB"]
        subprocess_return = subprocess.run(cmd, stdout=subprocess.DEVNULL)
        if subprocess_return.returncode != 0:
            raise Exception(f"Error running mmseqs2: {subprocess_return}")

    # 1.Additional input:
    # Representatives: mmseqs profile2repseq (here:in)profileDB (here:out)representativesDB
    # OR:
    # Consensus: mmseqs profile2consensus (here:in)profileDB (here:out)representativesDB
    
    # 2. Cluster profiles:
    # mmseqs search (here:in)profileDB (here:in)representativesDB (here:out)profileSearchDB tmp --add-self-matches (calculates distances)
    # mmseqs clust (here:in)profileDB (here:in)profileSearchDB (here:out)profileCluster --cluster-mode 1 --similarity-type 1

    def profile_search(self):
        cmd = ['mmseqs', 'search',
            self.profileDB,
            self.representativesDB,
            self.profilesearchDB,
            self.outPC / "tmp",
            '--add-self-matches',
            '-e','1e-5',
            '-s','7',
            '-c','0.90',
            '--cov-mode','0',
            '--qid','0.30']
        subprocess_return = subprocess.run(cmd, stdout=subprocess.DEVNULL)
        if subprocess_return.returncode != 0:
            raise Exception(f"Error running mmseqs2: {subprocess_return}")

    def profile_cluster(self):
        cmd = ['mmseqs', 'clust',
            self.profileDB,
            self.profilesearchDB,
            self.profileCluster,
            '--cluster-mode', '0']
        subprocess_return = subprocess.run(cmd, stdout=subprocess.DEVNULL)
        if subprocess_return.returncode != 0:
            raise Exception(f"Error running mmseqs2: {subprocess_return}")
    
    def parsePC(self):
        cmd = ['mmseqs', 'createtsv', self.profileDB, self.profileDB, self.profileCluster, self.outPC / "profileCluster.tsv"]
        subprocess_return = subprocess.run(cmd, stdout=subprocess.DEVNULL)
        if subprocess_return.returncode != 0:
            raise Exception(f"Error running mmseqs2: {subprocess_return}")
        
    def createPCDB(self):
        cmd = ['mmseqs', 'createsubdb', self.profileCluster, self.profileDB, self.profileClusterDB]
        subprocess_return = subprocess.run(cmd, stdout=subprocess.DEVNULL)
        if subprocess_return.returncode != 0:
            raise Exception(f"Error running mmseqs2: {subprocess_return}")
    
    def pcounts(self):
        with open(self.outPC / "profileCluster.tsv") as f:
            counts = Counter([line.strip().split("\t")[0] for line in f])
        return counts
    
    def PCDBreps(self, min_size: int):
        counts = self.pcounts()
        representatives = []
        for cluster in counts:
            if counts[cluster] >= min_size:
                representatives.append(cluster)
        return representatives

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

class SMMseqs2:
    def __init__(self, input_fasta: Path, mmseqs_db: Path, output_dir: Path) -> None:
        self.input_fasta = input_fasta
        self.mmseqs_db = mmseqs_db
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.result_file = self.output_dir / "result.m8"

    def run_alignment(self, threads: int, evalue: float) -> None:
        # Define paths for the created database and alignment results
        query_db = self.output_dir / "query_db"
        align_db = self.output_dir / "align_db"
        besthit_db = self.output_dir / "besthit_db"
        allhits_file = self.output_dir / "all_results.m8"

        # Create a database from the input FASTA file
        createdb_command = ["mmseqs", "createdb", self.input_fasta, query_db]
        self._run_command(createdb_command, "Failed to create query database")

        # Run search workflow (includes prefilter)
        align_command = [
            "mmseqs", "search",
            query_db,
            self.mmseqs_db,
            align_db,
            self.output_dir,
            "--threads", str(threads),
            '-s', '5'
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
            self.result_file,
            "--format-output", "query,target,evalue,bits"
        ]
        self._run_command(convertalis_command, "Result conversion failed")

        # Get all hits too:
        convertalis_command_2 = [
            "mmseqs", "convertalis",
            query_db,
            self.mmseqs_db,
            align_db,
            allhits_file,
            "--format-output", "query,target,evalue,bits"
        ]
        self._run_command(convertalis_command_2, "Result conversion failed")
        print(f"Alignment and result conversion completed. Results are in: {self.result_file}")
    
    def _run_command(self, command: list, error_message: str) -> None:
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            raise Exception(f"{error_message}: {e}")
        
    def get_best_hits(self):
        hits = {}
        with open(self.result_file) as f:
            for line in f:
                line = line.strip().split("\t")
                if line[0] not in hits:
                    hits[line[0]] = Hit(target_name=line[1], target_accession=line[1].replace('|','_'), bitscore=line[3])
                else:
                    if float(line[3]) > float(hits[line[0]].bitscore):
                        hits[line[0]] = Hit(target_name=line[1], target_accession=line[1].replace('|','_'), bitscore=line[3])
        return hits

class HHSUITE:
    def __init__(self, input: Path, out: Path):
        self.input = input
        self.out = out
    def convert(self, in_format, out_format):
        cmd = ['reformat.pl', in_format, out_format, '-i', self.input, '-o', self.out]
        subprocess_return = subprocess.run(cmd)
        if subprocess_return.returncode != 0:
            raise Exception(f"Error running hhconvert: {subprocess_return}")
    def make(self):
        cmd = ['hhmake', '-i', self.input, '-o', self.out]
        subprocess_return = subprocess.run(cmd)
        if subprocess_return.returncode != 0:
            raise Exception(f"Error running hhmake: {subprocess_return}")


def cat_alignments(input: Path, output: Path, rmdesc: bool = False):
    # Concat all alignments into one file:
    alignments = list(input.glob("*.st"))
    with open(output, "w") as f_out:
        for alignment in alignments:
            with open(alignment) as f_in:
                if rmdesc:
                    for line in f_in:
                        if not line.startswith("#="):
                            print(line.strip(), file=f_out)
                else:
                    for line in f_in:
                        print(line.strip(), file=f_out)
            print("\n", file=f_out)

def convert_alignments(msa: Path, output: Path):
    # Convert to msaDB, then profileDB, then index:
    msa_root = output / "mmseqsmsaDB"
    profile_root = output / "mmseqsprofileDB"
    msa_root.mkdir(parents=True, exist_ok=True)
    profile_root.mkdir(parents=True, exist_ok=True)
    logging.info("Converting alignments to msaDB...")
    cmd = ['mmseqs', 'convertmsa', msa, msa_root / "msaDB"]
    subprocess_return = subprocess.run(cmd, stdout=subprocess.DEVNULL)
    if subprocess_return.returncode != 0:
        raise Exception(f"Error running mmseqs convertmsa: {subprocess_return}")
    logging.info("Converting alignments to profileDB...")
    cmd = ['mmseqs', 'msa2profile', msa_root / "msaDB", profile_root / "profileDB", '--match-mode', '0', '--cov', '0.9', '--qid', '0.5']
    subprocess_return = subprocess.run(cmd, stdout=subprocess.DEVNULL)
    if subprocess_return.returncode != 0:
        raise Exception(f"Error running mmseqs msa2profile: {subprocess_return}")
    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        logging.info("Indexing profileDB..")
        cmd = ['mmseqs', 'createindex', '-s', '6', profile_root / "profileDB", tmpdir]
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
    output = Path("data/processed/10_datasets/attachings_blitsDB")
    output_path = dataset / "strings" / "blitsDB"
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
    A3M_CONVERSION = False

    # After HHBlits
    STO_CONVERSION = False
    CONVERT_ALIGNMENTS = False
    MMSEARCH_REPRESENTATIVES = False

    # Cluster profiles
    PROFILE_CLUSTER = False
    MMRESEARCH_REPRESENTATIVES = True

    TRANSLATE_DATASET = False
    SCAN_DATASET = True
    ATTACH_TO_DATASET = True

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
        cat_alignments(alignment_root, output / "alignments.st")
        logging.info("Done aligning clusters.")

    if A3M_CONVERSION:
        logging.info("Converting alignments to A3M...")
        a3m_root = output / "alignments_a3m"
        a3m_root.mkdir(parents=True, exist_ok=True)
        for representative in tqdm(representatives):
            clu_input = output / "alignments" / f"{representative.replace('|','_')}.st"
            a3m_output = a3m_root / f"{representative.replace('|','_')}.a3m"
            hh = HHSUITE(input=clu_input, out=a3m_output)
            hh.convert(in_format="sto", out_format="a3m")

    # (optional v1 enrichment) reformat stockholm files to a3m
    # (optional v1 enrichment) run hhblits with a3m files
    # (optional v1 enrichment) reformat a3m files to stockholm

    # (optional v2 enrichment) bypass alignment extract individual fasta files
    # (optional v2 enrichment) run colabfold search with fasta files
    # (optional v2 enrichment) reformat a3m files to stockholm

    ####
    ## Manual remove (iteration)
    ####

    unused = {'PS_U1976|25', 'PS_U121|1', 'PS_U909|15', 'PS_R2278|14', 'PS_U2368|15'}

    if STO_CONVERSION:
        logging.info("Converting alignments from a3m to STO...")
        alignment_root = output / "alignments_sto"
        alignment_root.mkdir(parents=True, exist_ok=True)
        for representative in tqdm(representatives):
            if representative in unused:
                continue
            a3m_input = output / "alignments_a3m" / f"{representative.replace('|','_')}.a3m"
            sto_output = alignment_root / f"{representative.replace('|','_')}.st"
            hh = HHSUITE(input=a3m_input, out=sto_output)
            hh.convert(in_format="a3m", out_format="sto")
        cat_alignments(alignment_root, output / "alignments.st", rmdesc=True)
    
    if CONVERT_ALIGNMENTS:
        # Convert to msaDB, then profileDB, then index:
        convert_alignments(msa=output / "alignments.st", output=output)
        logging.info("Done converting alignments to profileDB.")

    smm = SMMseqs2(input_fasta=output / "representatives.faa", mmseqs_db=output / "mmseqsprofileDB" / "profileDB", output_dir=output / "mmsearch")
    
    if MMSEARCH_REPRESENTATIVES:
        logging.info("Searching representatives...")
        smm.run_alignment(threads=num_threads, evalue=1e-3)
        logging.info("Done searching representatives.")

    ## 
    # 1.Additional input:
    # mmseqs createsubdb (here:in)clusterDB (here:out)representativesDB (extract original representatives)

    # 2. Cluster profiles:
    # mmseqs search (here:in)profileDB (here:in)representativesDB (here:out)profileSearchDB tmp -e 0.05 --add-self-matches (calculates distances)
    # mmseqs clust (here:in)profileDB (here:in)profileSearchDB profileClusterDB--cluster-mode 1 --similarity-type 1
    
    mm.prep_pcDB(outf=output / "mmseqsprofileclusterDB", profileDB=output / "mmseqsprofileDB" / "profileDB")

    if PROFILE_CLUSTER:
        logging.info("Extracting representatives...")
        mm.create_repDB()
        logging.info("Profiles search...")
        mm.profile_search()
        logging.info("Profile cluster...")
        mm.profile_cluster()
        logging.info("Parsing profile cluster...")
        mm.parsePC()
        logging.info("Creating profile cluster DB...")
        mm.createPCDB()
        logging.info("Done creating profile cluster DB.")


    rmm = SMMseqs2(input_fasta=output / "representatives.faa", mmseqs_db=output / "mmseqsprofileclusterDB" / "profileclusterDB", output_dir=output / "mmresearch")
    if MMRESEARCH_REPRESENTATIVES:
        logging.info("Searching representatives...")
        rmm.run_alignment(threads=num_threads, evalue=1e-3)
        logging.info("Done searching representatives.")
    
    profile_reps = mm.PCDBreps(min_size=0)
    hit_reps = rmm.get_best_hits()
    representatives_used = set([hit_reps[rep].target_accession for rep in hit_reps])
    reps = set([i.replace('|','_') for i in profile_reps])
    logging.info(f"Number of representatives: {len(reps)}")
    logging.info(f"Number of representatives used: {len(representatives_used)}")
    logging.info(f"Number of representatives not used: {len(reps - representatives_used)}")
    logging.info("Representatives not found:")
    for rep in set(representatives) - set(hit_reps.keys()):
        print(rep)
    logging.info(f"Number of representatives not found: {len(set(representatives) - set(hit_reps.keys()))}")


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
        out_path = output / f"mm_{split}_search"
        out_path.mkdir(parents=True, exist_ok=True)

    mm_splits = { split: SMMseqs2(input_fasta=output / f"{split}_split.faa", mmseqs_db=output / "mmseqsprofileclusterDB" / "profileclusterDB", output_dir=output / f"mm_{split}_search") for split in splits}

    if SCAN_DATASET:
        logging.info("Scanning dataset...")
        for split in splits:
            print(f"Split: {split}")
            mm_splits[split].run_alignment(threads=num_threads, evalue=1e-3)
        logging.info("Done scanning dataset.")

    if ATTACH_TO_DATASET:
        logging.info("Attaching dataset...")
        hmm_data_splits = torch.load(output / "translated.pt")
        for split in splits:
            print(f"Split: {split}")
            hits = mm_splits[split].get_best_hits()
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
        accessions = set([i.replace('|','_') for i in profile_reps])
        print(f"Number of non-redundant accessions in HMM database: {len(accessions)}")
        print(f"Number of non-redundant accessions used in dataset: {len(vocab)}")
        print(f"Accessions numbers not used: {len(accessions - vocab)}")
        print(f"Accessions numbers not used: {accessions - vocab}")
        torch.save(hmm_data_splits, output_path / "dataset.pt")
        torch.save(vocab_map, output_path / "vocab_map.pt")

        logging.info("Done attaching dataset.")
