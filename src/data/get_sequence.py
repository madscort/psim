import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pyrodigal
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import shutil
import click
import logging
import subprocess
import sys
import collections
import pandas as pd
import tempfile
from pyhmmer import easel, plan7, hmmsearch, hmmscan
from src.models.set_cover import sc_greedy
from src.data.get_hmms import get_hmms


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


def get_pan_proteins(protein_seqs: Path = Path("data/raw/proteins/all_proteins.fna"),
                     cluster_mode: int = 1,
                     cov_mode: int = 0,
                     min_cov: float = 0.9,
                     min_seq_id: float = 0.3,
                     rerun_all: bool = False,
                     get_topx: int = None,
                     out_prefix: str = "cluster3090DB",
                     out_file_root: Path = Path("data/processed/protein_clusters/test/cluster3090DB")):

    rerun_all = rerun_all
    get_topx = get_topx
    
    cluster_mode = cluster_mode
    cov_mode = cov_mode
    min_cov = min_cov
    min_seq_id = min_seq_id

    mmseqs_tmp = Path("./.tmp/mmseqs")
    mmseqs_tmp.mkdir(parents=True, exist_ok=True)
    protein_seqs = protein_seqs

    out_prefix = out_prefix
    out_file_root = out_file_root

    proteinsDB = Path(out_file_root, "mmseqsDB", "proteinDB")
    mmseq_clusters = Path(out_file_root, "cluster", out_prefix)
    rep_protein_seqs = Path(out_file_root, "representatives", out_prefix).with_suffix(".fna")

    proteinsDB.parent.mkdir(parents=True, exist_ok=True)
    mmseq_clusters.parent.mkdir(parents=True, exist_ok=True)
    rep_protein_seqs.parent.mkdir(parents=True, exist_ok=True)

    """Create protein database for clustering with MMseqs2
    """

    if proteinsDB.exists() and not rerun_all:
        logging.info(f"Skipping mmseqs2 createdb, {proteinsDB.absolute().as_posix()} already exists.")
    else:
        proteinsDB.unlink(missing_ok=True)
        proteinsDB.parent.mkdir(parents=True, exist_ok=True)

        logging.info(f"Running mmseqs2 createdb on {proteinsDB.absolute().as_posix()}")

        cmd = ["mmseqs", "createdb", protein_seqs, proteinsDB]
        subprocess_return = subprocess.run(cmd, stdout=subprocess.DEVNULL)
        if subprocess_return.returncode != 0:
            raise Exception(f"Error running mmseqs2: {subprocess_return}")
    
    """Cluster proteins with MMseqs2
    """

    if Path(mmseq_clusters).with_suffix(".1").exists() and not rerun_all:
        logging.info(f"Skipping mmseqs2 cluster, {mmseq_clusters.absolute().as_posix()} already exists.")
    else:
        for ps in mmseq_clusters.parent.glob(str(mmseq_clusters.name) + "*"):
            ps.unlink(missing_ok=True)
        
        mmseq_clusters.parent.mkdir(parents=True, exist_ok=True)
        mmseqs_tmp.mkdir(parents=True, exist_ok=True)

        logging.info(f"Running mmseqs2 cluster on {mmseq_clusters.absolute().as_posix()}")

        cmd = ['mmseqs','cluster', '--cluster-mode', f"{cluster_mode}", '--min-seq-id', f"{min_seq_id}", '--cov-mode', f"{cov_mode}",
               '-c', f"{min_cov}", proteinsDB, mmseq_clusters, mmseqs_tmp]
        subprocess_return = subprocess.run(cmd)
        if subprocess_return.returncode != 0:
            raise Exception(f"Error running mmseqs2: {subprocess_return}")

    """Create tsv file with cluster and representative protein
    """

    if Path(str(mmseq_clusters)+'.tsv').exists() and not rerun_all:
        logging.info(f"Skipping mmseqs2 createtsv, {mmseq_clusters.absolute().as_posix()} already exists.")
    else:
        logging.info(f"Running mmseqs2 createtsv on {mmseq_clusters.absolute().as_posix()}")
        cmd = ['mmseqs','createtsv', proteinsDB, proteinsDB, mmseq_clusters, Path(str(mmseq_clusters)+'.tsv')]
        subprocess_return = subprocess.run(cmd, stdout=subprocess.DEVNULL)
        if subprocess_return.returncode != 0:
            raise Exception(f"Error running mmseqs2: {subprocess_return}")

    if get_topx == None:
        if rep_protein_seqs.exists() and not rerun_all:
            logging.info(f"Skipping set cover, {mmseq_clusters.absolute().as_posix()} already exists.")
        else:
            logging.info(f"Running set cover on {mmseq_clusters.absolute().as_posix()}")
            ordered_cluster_reps, _ = sc_greedy(mmseq_clusters = Path(str(mmseq_clusters)+'.tsv'))
            logging.info(f"Number of cluster representatives: {len(ordered_cluster_reps)}")
            records = get_records_from_identifiers(fasta=protein_seqs, identifiers=ordered_cluster_reps)
            SeqIO.write(records, rep_protein_seqs.absolute().as_posix(), "fasta")
        return rep_protein_seqs
    else:
        with open(Path(str(mmseq_clusters)+'.tsv'), 'r') as f:
            # Get largest x clusters:
            df = pd.read_csv(f, sep='\t', header=None, names=['cluster', 'protein'])
            df = df.groupby('cluster').count().sort_values(by='protein', ascending=False).head(get_topx).reset_index()
            topx_cluster_reps = df['cluster'].values.tolist()
            records = get_records_from_identifiers(fasta=protein_seqs, identifiers=topx_cluster_reps)
            SeqIO.write(records, rep_protein_seqs.absolute().as_posix(), "fasta")
        return rep_protein_seqs


def get_gene_gc_sequence(fna_file: Path = Path("data/processed/10_datasets/phage_25_reduced_90/sequences/PS_U370.fna")):
    """ Takes a fasta file and returns a list of GC contents for each gene."""

    input_sequence_file = fna_file
    output = list()
    pyro = pyrodigal.OrfFinder(meta=True)
    
    with open(input_sequence_file, 'r') as f:
        contig = SeqIO.parse(f, "fasta").__next__()
        for n, gene in enumerate(pyro.find_genes(bytes(contig.seq))):
            output.append(gene.gc_cont)
    return output

def get_protein_seq_str(fna_file: Path) -> [str]:
    """ Takes a fna file and returns a list of proteins as strings."""

    pyro = pyrodigal.GeneFinder(meta=True)
    proteins = list()
    with open(fna_file, 'r') as f:
        for sequence in SeqIO.parse(f, "fasta"):
            for n, gene in enumerate(pyro.find_genes(bytes(sequence.seq))):
                proteins.append(gene.translate(include_stop=False))
    return proteins

def get_protein_seq(fna_files: [Path], faa_file: Path) -> Path:
    """ Takes a fna file and saves all proteins. Returns path to faa file."""

    pyro = pyrodigal.OrfFinder(meta=True)

    with open(faa_file, 'w') as faa:
        for input_sequence_file in fna_files:
            with open(input_sequence_file, 'r') as f:
                for sequence in SeqIO.parse(f, "fasta"):
                    for n, gene in enumerate(pyro.find_genes(bytes(sequence.seq))):
                        SeqIO.write(SeqRecord(Seq(gene.translate()),
                                    id=f"{sequence.id}|Protein_{n}",
                                    description=""), faa, format="fasta")
    return faa_file

def get_pyhmms(input_faa: Path = Path("models/hmm_model/phage_25_reduced_90/pan_proteins/representatives/phage_25_reduced_90.fna"), output_hmm: Path = Path("./.tmp/test.hmm"), hmm_db: Path = Path("data/external/databases/pfam/pfam_A/Pfam-A.hmm")):
    """ Takes a fna file and returns """
    Result = collections.namedtuple("Result", ["query", "pfam", "bitscore"])
    Output = collections.namedtuple("Output", ["accession", "name", "description"])
    results = {}
    with plan7.HMMFile(hmm_db) as hmms:
        with easel.SequenceFile(input_faa, digital=True) as seqs:
            for hits in hmmsearch(hmms, seqs, cpus=1, E=0.001):
                for hit in hits.included:
                    if hit.name.decode() not in results:
                        results[hit.name.decode()] = Result(hit.name.decode(), hits.query_accession.decode(), hit.score)
                    else:
                        if hit.score > results[hit.name.decode()].bitscore:
                            results[hit.name.decode()] = Result(hit.name.decode(), hits.query_accession.decode(), hit.score)
                    

    pfams_to_keep = set([results[k].pfam for k in results])
    hmm_info = []
    # HMMs to a new file
    with open(output_hmm, 'wb') as new_hmm_file:
        with plan7.HMMFile(hmm_db) as hmms:
            for hmm in hmms:
                if hmm.accession.decode() in pfams_to_keep:
                    hmm_info.append(Output(hmm.accession.decode(), hmm.name.decode(), hmm.description.decode()))
                    hmm.write(new_hmm_file)
    for info in hmm_info:
        print(info.accession, info.name, info.description)
    return output_hmm, hmm_info

def get_marker_hmms(fna_files: [Path] = [Path("data/processed/10_datasets/phage_25_reduced_90/sequences/PS_U370.fna")],
                 tmp_folder: Path = Path("./.tmp/cluster_seq_creation"),
                 get_topx: int = 10):
    """ Takes a list of fna files and returns a hmm file. """
    logging.info(f"Marker build - Translating {len(fna_files)} files")
    proteins = get_protein_seq(fna_files, tmp_folder / "proteins.faa")
    logging.info(f"Marker build - Clustering")
    clusters = get_pan_proteins(protein_seqs=proteins, out_file_root=tmp_folder, get_topx=get_topx)
    logging.info(f"Marker build - Finding HMMs")
    hmms = get_hmms(rerun=True, rep_protein_seqs=clusters, out_file_root=tmp_folder)
    return hmms

def scan_hmms(input_faa: Path = Path("data/processed/10_datasets/phage_25_reduced_90/sequences/PS_U370.fna"), hmm_db: Path = Path(".tmp/cluster_seq_creation/cluster3090_hmm_profiles.hmm")):
    """ Utility for get_marker_match_sequence. Takes a faa file and returns a list of hmm matches. """
    results = {}
    with plan7.HMMFile(hmm_db) as hmms:
        with easel.SequenceFile(input_faa, digital=True) as seqs:
            for hits in hmmscan(seqs, hmms, cpus=1):
                results[hits.query_name.decode()] = "no_hit"
                max_score = 0
                for hit in hits:
                    if hit.score > max_score:
                        max_score = hit.score
                        results[hits.query_name.decode()] = hit.accession.decode()
    return [results[k] for k in results]


def get_marker_match_sequence(fna_file: Path = Path("data/processed/10_datasets/phage_25_reduced_90/sequences/PS_U370.fna"),
                              hmm: Path = Path(".tmp/cluster_seq_creation/cluster3090_hmm_profiles.hmm")):
    """ Takes a single fna file and returns a sequence (list) of hmm matches """
    with tempfile.TemporaryDirectory() as tmp_work:
        proteins = get_protein_seq([fna_file], Path(tmp_work) / "proteins.faa")
        sequence_of_hmms = scan_hmms(proteins, hmm_db=hmm)
    return sequence_of_hmms

from sklearn.model_selection import train_test_split
import torch

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    dataset_root = Path("data/processed/10_datasets/phage_25_fixed_25000_reduced_90_ws")
    sampletable = Path("data/processed/10_datasets/phage_25_fixed_25000_reduced_90_ws/sampletable.tsv")
    datafolder = Path("data/processed/10_datasets/phage_25_fixed_25000_reduced_90_ws/sequences")
    psfolder = Path("data/processed/10_datasets/phage_25_fixed_25000_reduced_90_ws/satellite_sequences")
    df_sampletable = pd.read_csv(sampletable, sep="\t", header=None, names=['id', 'type', 'label'])

    fit, test = train_test_split(df_sampletable, stratify=df_sampletable['type'], test_size=0.1)
    train, val = train_test_split(fit, stratify=fit['type'], test_size=0.2)

    train_sequences = [Path(datafolder, f"{id}.fna") for id in train['id'].values]
    val_sequences = [Path(datafolder, f"{id}.fna") for id in val['id'].values]
    test_sequences = [Path(datafolder, f"{id}.fna") for id in test['id'].values]
    ps_files = [Path(psfolder, f"{id}.fna") for id in train[train['label'] == 1]['id'].values]

    use_saved = True

    if use_saved and Path(dataset_root, "hmm_matches.pt").exists():
        with open(Path(dataset_root, "hmm_matches.pt"), "rb") as f:
            hmm_matches = torch.load(f)
    else:
        pass
        # with tempfile.TemporaryDirectory() as tmp_work:
        #     # Get HMMs from phage satellite sequences only
        #     hmm = get_marker_hmms(fna_files=ps_files,
        #                         tmp_folder=Path(tmp_work), get_topx=500)
        #     hmm_matches = {}
        #     hmm_matches['train'] = [get_marker_match_sequence(fna_file=fna_file, hmm=hmm) for fna_file in train_sequences]
        #     hmm_matches['val'] = [get_marker_match_sequence(fna_file=fna_file, hmm=hmm) for fna_file in val_sequences]
        #     hmm_matches['test'] = [get_marker_match_sequence(fna_file=fna_file, hmm=hmm) for fna_file in test_sequences]
            
        #     torch.save(hmm_matches, Path(dataset_root, "hmm_matches.pt"))
    for sample in hmm_matches['train']:
        for seq in sample:
            print(seq)
    # with tempfile.TemporaryDirectory() as tmp_work:
    #     hmm = get_marker_hmms(fna_files=[Path("data/processed/01_combined_renamed/reduced_90/all_sequences.fna")],
    #                           tmp_folder=Path(tmp_work),
    #                           get_topx=500)
    #     print("hmm file:", hmm)
    #     match_seq = get_marker_match_sequence(fna_file=Path("data/processed/10_datasets/phage_25_reduced_90/sequences/PS_U370.fna"))
    #     for seq in match_seq:
    #         print(seq)
