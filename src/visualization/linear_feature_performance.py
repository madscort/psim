import shap
import numpy as np
import pandas as pd
import logging
import sys
import matplotlib.pyplot as plt
from Bio import SeqIO
from torch import load as tload
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.models.LNSequenceModule import SequenceModule

def encode_sequence(sequence, vocabulary_mapping):
    return [vocabulary_mapping[gene_name] for gene_name in sequence]

def encode_wo_nohit_sequence(sequence, vocabulary_mapping):
    return [vocabulary_mapping[gene_name] for gene_name in sequence if gene_name != "no_hit"]

def count_encode(samples, vocab_size):
    features = np.zeros((len(samples), vocab_size), dtype=int)
    for i, sample in enumerate(samples):
        for gene_code in sample:
            if not gene_code-1 < vocab_size:
                print("ERROR: ", gene_code)
            else:
                features[i, gene_code-1] += 1
    
    return features

def main():

    root = Path("/Users/madsniels/Documents/_DTU/speciale/cpr/code/psim")
    dataset_root = root / "data/processed/10_datasets/"
    version = "dataset_v02"
    dataset = dataset_root / version
    checkpoint = root / "models/transformer/alldb_v02_small_iak7l6eg.ckpt"
    output_root = root / "data/visualization/performance/linear_feature_selection/"
    output_root.mkdir(parents=True, exist_ok=True)
    outputfn = output_root / "logistic_alldb_v02_feature_selection_test.txt"
    out_f = Path("data/visualization/performance/linear_feature_selection/feature_representatives/")
    out_f.mkdir(parents=True, exist_ok=True)
    in_fasta = Path("data/processed/10_datasets/v02/attachings_allDB/proteins.faa").absolute()

    mo = SequenceModule.load_from_checkpoint(checkpoint_path=checkpoint, map_location="cpu")
    vocab_map = mo.vocab_map
    del vocab_map['<PAD>']
    del vocab_map['no_hit']

    scaler = StandardScaler()

    data_splits = tload(dataset / "strings" / "allDB" / "dataset.pt", map_location="cpu")
    X_train = [encode_wo_nohit_sequence(string, vocab_map) for string in data_splits['train']['sequences']]
    X_train = count_encode(X_train, len(vocab_map))
    X_train = scaler.fit_transform(X_train)
    y_train = data_splits['train']['labels'].numpy()
    
    X_val = [encode_wo_nohit_sequence(string, vocab_map) for string in data_splits['test']['sequences']]
    X_val = count_encode(X_val, len(vocab_map))
    X_val = scaler.transform(X_val)
    y_val = data_splits['test']['labels'].numpy()

    c_values = [10]
    with open(outputfn, "w") as fout:
        for n, c in enumerate(c_values):
            model = LogisticRegression(penalty='l1', solver='saga', C=c, max_iter=5000, random_state=1, n_jobs=6)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred)
            recall = recall_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred)
            roc_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
            coefficients = model.coef_[0]
            pos_coefficients = coefficients[coefficients > 0]
            neg_coefficients = coefficients[coefficients < 0]
            tot_coeff = len(pos_coefficients) + len(neg_coefficients)
            logging.info(f"C: {c}, Features: {tot_coeff}, Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
            metrics = {"accuracy": accuracy,
                       "precision": precision,
                       "recall": recall,
                       "f1": f1,
                       "roc_auc": roc_auc}
            
            for metric in metrics:
                print(n,
                      c,
                      len(pos_coefficients),
                      len(neg_coefficients),
                      metric,
                      metrics[metric],
                      sep="\t",
                      file=fout)

            out_fasta = out_f / f"c{c}_sequences.faa"
            feature_names = list(vocab_map.keys())
            feature_importance = dict(zip(feature_names, coefficients))
            sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
            with open(out_fasta, "w") as fasta:
                for n, (feature, coef) in enumerate(sorted_features):
                    if coef == 0:
                        continue
                    else:
                        for seq_record in SeqIO.parse(in_fasta, "fasta"):
                            if seq_record.id.replace('|','_') == feature:
                                SeqIO.write(seq_record, fasta, "fasta")
                                break
                        else:
                            print(f"ERROR: {feature} not found in fasta file")

if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
