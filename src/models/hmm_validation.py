
import logging
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src.data.get_pan_proteins import get_pan_proteins
from src.data.get_hmms import get_hmms
from src.models.hmm_predict import hmm_predict
from Bio import SeqIO

def main():
    
    rerun = False

    version_id = "phage_25_reduced_90_combined"

    cluster_mode = 1
    cov_mode = 0
    min_cov = 0.8
    min_seq_id = 0.4

    phmm_database = Path("data/external/databases/pfam/combined/combined.hmm")
    
    # Possible database:
    # pfamA: Path("data/external/databases/pfam/pfam_A/Pfam-A.hmm")
    # vfam: Path("data/external/databases/pfam/vfam_A/vFam-A_2014.hmm")
    # RVDB: Path("data/external/databases/pfam/RVDB/U-RVDBv26.0-prot.hmm")
    # combined pfamA+RVDB: Path("data/external/databases/pfam/combined/combined.hmm")
    
    pfamA = False
    dataset = Path("data/processed/10_datasets/phage_25_reduced_90")
    sample_table = Path(dataset, "sampletable.tsv")
    all_sequences = Path(dataset, "dataset.fna")
    positive_protein_collection = Path("data/processed/01_combined_renamed/all_proteins.faa")

    validation_root = Path("models/hmm_model")
    iteration_root = Path(validation_root, version_id)
    tmp = Path(iteration_root, ".tmp")
    tmp.mkdir(parents=True, exist_ok=True)

    train_pos_fasta = Path(tmp, "train_pos.fna")
    test_pos_fasta = Path(tmp, "test_pos.faa")
    test_neg_fasta = Path(tmp, "test_neg.faa")

    logging.info("Splitting data into training and test sets")

    # Split positive and negetative sets separately
    df_sampletable = pd.read_csv(sample_table, sep="\t", header=None, names=['id', 'type', 'label'])

    df_neg = df_sampletable.loc[df_sampletable['label'] == 0]
    df_pos = df_sampletable.loc[df_sampletable['label'] == 1]

    _, test_neg = train_test_split(df_neg, stratify=df_neg['type'], test_size=0.2)
    train_pos, test_pos = train_test_split(df_pos, stratify=df_pos['type'], test_size=0.2)

    # Create temporary split datasets

    # Get proteins for positive samples for training set

    logging.info("Creating temporary fasta files for training and test sets")

    if train_pos_fasta.exists() and not rerun:
        logging.info(f"Skipping train_pos_fasta, {train_pos_fasta.absolute().as_posix()} already exists.")
    else:
        logging.info(f"Creating {train_pos_fasta.absolute().as_posix()}")
        train_pos_fasta.unlink(missing_ok=True)
        with open(train_pos_fasta, "w") as f:
            for record in SeqIO.parse(positive_protein_collection, "fasta"):
                ps_name = record.id.split("|")[0]
                if ps_name in train_pos['id'].values:
                    SeqIO.write(record, f, "fasta")

    if not test_pos_fasta.exists() or not test_neg_fasta.exists() or rerun:
        sequence_index = SeqIO.index(all_sequences.absolute().as_posix(), "fasta")

    if test_pos_fasta.exists() and test_neg_fasta.exists() and not rerun:
        logging.info(f"Skipping test_pos_fasta, {test_pos_fasta.absolute().as_posix()} already exists.")
    else:
        logging.info(f"Creating {test_pos_fasta.absolute().as_posix()}")
        test_pos_fasta.unlink(missing_ok=True)

        with open(test_pos_fasta, "w") as f:
            for id in test_pos['id'].values:
                SeqIO.write(sequence_index[id], f, "fasta")
    
    if test_neg_fasta.exists() and not rerun:
        logging.info(f"Skipping test_neg_fasta, {test_neg_fasta.absolute().as_posix()} already exists.")
    else:
        logging.info(f"Creating {test_neg_fasta.absolute().as_posix()}")
        test_neg_fasta.unlink(missing_ok=True)
        with open(test_neg_fasta, "w") as f_out:  
            for id in test_neg['id'].values:
                SeqIO.write(sequence_index[id], f_out, "fasta")

    try:
        sequence_index.close()
    except:
        pass

    ### Training ###
    # Create hmms with training set
    # Defined clustering types etc.    

    # Run get_pan_proteins function:

    rep_seq = get_pan_proteins(protein_seqs = train_pos_fasta,
                               cluster_mode = cluster_mode,
                               cov_mode = cov_mode,
                               min_cov = min_cov,
                               min_seq_id = min_seq_id,
                               rerun_all = rerun,
                               out_prefix = version_id,
                               out_file_root = Path(iteration_root, "pan_proteins"))

    # Run get_hmms function:

    hmm_profiles = get_hmms(rerun = rerun,
                            pfamA=pfamA,
                            pfam_hmm = phmm_database,
                            re_index_pfam = False,
                            rep_protein_seqs = rep_seq,
                            out_prefix = version_id,
                            out_file_root = Path(iteration_root, "hmm_profiles"))

    ### Test ###
    # Run model on positive set

    if Path(iteration_root, "prediction", "positive", f"{version_id}_pos_hmmsearch_result.tsv").exists() and not rerun:
        logging.info(f"Skipping prediction on positive set")
        with open(Path(iteration_root, "prediction", "positive", f"{version_id}_pos_hmmsearch_result.tsv")) as f:
            positive_results = []
            for line in f:
                res = [float(res) for res in line.strip().split("\t")]
                positive_results.append(res)
    else:
        logging.info(f"Running prediction on positive set")
        positive_results = hmm_predict(input_contigs = test_pos_fasta,
                                       hmm_profiles = hmm_profiles,
                                       out_prefix = f"{version_id}_pos",
                                       out_file_root = Path(iteration_root, "prediction", "positive"),
                                       threshold = None)
    
    # Run model on negative set

    if Path(iteration_root, "prediction", "negative", f"{version_id}_neg_hmmsearch_result.tsv").exists() and not rerun:
        logging.info(f"Skipping prediction on negative set")
        with open(Path(iteration_root, "prediction", "negative", f"{version_id}_neg_hmmsearch_result.tsv")) as f:
            negative_results = []
            for line in f:
                res = [float(res) for res in line.strip().split("\t")]
                negative_results.append(res)
    else:
        logging.info(f"Running prediction on negative set")
        negative_results = hmm_predict(input_contigs = test_neg_fasta,
                                       hmm_profiles = hmm_profiles,
                                       out_prefix = f"{version_id}_neg",
                                       out_file_root = Path(iteration_root, "prediction", "negative"),
                                       threshold = None)

    # Zip bit scores and create labels
    # Max version:
    max_positive = [max(res) for res in positive_results]
    max_negative = [max(res) for res in negative_results]
    max_scores = max_positive + max_negative
    max_labels = [1]*len(max_positive) + [0]*len(max_negative)

    # Get roc curve with scikit learn

    fpr, tpr, thresholds = roc_curve(max_labels, max_scores)
    roc_auc = auc(fpr, tpr)

    # Plot using matplotlib

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    Path(iteration_root, "result_metrics", "roc_curve.png").parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(iteration_root, "result_metrics", "roc_curve.png"))
    plt.close()
    # Get precision recall curve with scikit learn

    precision, recall, thresholds = precision_recall_curve(max_labels, max_scores)
    pr_auc = auc(recall, precision)

    # Plot using matplotlib
    plt.figure()
    plt.plot(recall, precision, color='darkorange', lw=2, label='PR curve (area = %0.2f)' % pr_auc)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall Curve')
    plt.legend(loc="lower right")
    Path(iteration_root, "result_metrics", "pr_curve.png").parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(iteration_root, "result_metrics", "pr_curve.png"))

    # Get precision recall and f1 score with scikit learn
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal threshold: {optimal_threshold}")

    precision, recall, f1, _ = precision_recall_fscore_support(max_labels, [1 if score > optimal_threshold else 0 for score in max_scores], average='binary')

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")

    # Save results to file
    with open(Path(iteration_root, "result_metrics", "metrics.tsv"), "w") as f:
        print(f"Optimal threshold: {optimal_threshold}", file = f)
        print(f"Precision: {precision}", file = f)
        print(f"Recall: {recall}", file = f)
        print(f"F1: {f1}", file = f)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
