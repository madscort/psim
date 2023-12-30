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
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from scipy.stats import t


rename = {'test_acc': 'accuracy_score', 'test_f1': 'f1_score', 'test_precision': 'precision_score', 'test_auc': 'roc_auc_score', 'test_recall': 'recall_score'}

def calculate_means_and_cis_to_file(metrics_list, file_path):
    # Initialize a dictionary to hold our aggregated metrics
    aggregated_metrics = {}
    
    # Loop through each metric in the first fold to initialize the keys and lists
    for key in metrics_list[0].keys():
        aggregated_metrics[key] = []

    # Now populate the lists with values from all folds
    for metrics in metrics_list:
        for key, value in metrics.items():
            aggregated_metrics[key].append(value)

    # Number of observations (folds)
    n = len(metrics_list)
    # Significance level for 95% confidence
    alpha = 0.05
    # Degrees of freedom
    df = n - 1
    # Critical t value for two tails
    t_crit = abs(t.ppf(alpha/2, df))

    # Calculate the mean and confidence intervals
    with open(file_path, 'w') as fout:
        for metric, values in aggregated_metrics.items():
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1)
            ci_half_width = t_crit * (std_val / np.sqrt(n))
            lower_ci = mean_val - ci_half_width
            upper_ci = mean_val + ci_half_width
            
            print('Baseline', rename[metric], mean_val, lower_ci, upper_ci, sep='\t', file=fout)

def custom_stratify(df, stratify_col, small_class_threshold=10):
    # Get small classes
    small_classes = df[stratify_col].value_counts()[df[stratify_col].value_counts() < small_class_threshold].index

    # Separate
    df_small = df[df[stratify_col].isin(small_classes)]
    df_large = df[~df[stratify_col].isin(small_classes)]

    # Split large classes with stratification
    train_idx_large, test_idx_large = train_test_split(df_large.index, stratify=df_large[stratify_col], test_size=0.1, random_state=1)
    # Randomly split small classes
    train_idx_small, test_idx_small = train_test_split(df_small.index, test_size=0.1, random_state=1)

    # Combine
    train_idx = train_idx_large.union(train_idx_small)
    test_idx = test_idx_large.union(test_idx_small)

    return train_idx, test_idx

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

def resplit_data(data_splits, train_indices, val_indices, test_indices):
    data_sequences = []
    data_labels = []
    for split in ['train', 'val', 'test']:
        data_sequences.extend(data_splits[split]['sequences'])
        data_labels.extend(data_splits[split]['labels'])

    data_splits = {
        'train': {'sequences': [data_sequences[i] for i in train_indices], 'labels': [data_labels[i] for i in train_indices]},
        'val': {'sequences': [data_sequences[i] for i in val_indices], 'labels': [data_labels[i] for i in val_indices]},
        'test': {'sequences': [data_sequences[i] for i in test_indices], 'labels': [data_labels[i] for i in test_indices]}
    }

    return data_splits

def main():

    root = Path("/Users/madsniels/Documents/_DTU/speciale/cpr/code/psim")
    dataset_root = root / "data/processed/10_datasets/"
    version = "dataset_v02"
    dataset = dataset_root / version
    checkpoint = root / "models/transformer/alldb_v02_small_iak7l6eg.ckpt"
    output_metrics = Path("data/visualization/performance/v02/confidence/cross_linear.tsv")
    output_metrics_raw = Path("data/visualization/performance/v02/confidence/cross_linear_raw.tsv")
    
    sampletables = [pd.read_csv(dataset / f"{split}.tsv", sep="\t", header=0, names=["id", "type", "label"]) for split in ['train', 'val', 'test']]
    df_sampletable = pd.concat(sampletables, ignore_index=True)

    c_value = 1
    max_it = 5000

    mo = SequenceModule.load_from_checkpoint(checkpoint_path=checkpoint, map_location="cpu")
    vocab_map = mo.vocab_map
    del vocab_map['<PAD>']
    del vocab_map['no_hit']

    scaler = StandardScaler()

    data_splits = tload(dataset / "strings" / "allDB" / "dataset.pt", map_location="cpu")
    
    # Initialize the StratifiedKFold object
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    fold_results = []
    for fold, (fit_idx, test_idx) in enumerate(skf.split(df_sampletable['id'], df_sampletable['type'])):
        print(f"Fold {fold}")
        # Split the training data into training and validation sets
        
        train_idx, val_idx = custom_stratify(df_sampletable.iloc[fit_idx], 'type')
        
        data_splits = resplit_data(data_splits=data_splits, train_indices=train_idx, val_indices=val_idx, test_indices=test_idx)

    
        X_train = [encode_wo_nohit_sequence(string, vocab_map) for string in data_splits['train']['sequences']]
        X_train = count_encode(X_train, len(vocab_map))
        X_train = scaler.fit_transform(X_train)
        y_train = data_splits['train']['labels']
        
        X_val = [encode_wo_nohit_sequence(string, vocab_map) for string in data_splits['test']['sequences']]
        X_val = count_encode(X_val, len(vocab_map))
        X_val = scaler.transform(X_val)
        y_val = data_splits['test']['labels']

        model = LogisticRegression(penalty='l1', solver='saga', C=c_value, max_iter=max_it, random_state=1, n_jobs=6)
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
        logging.info(f"Fold: {fold}, Features: {tot_coeff}, Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

        fold_results.append({'test_acc': accuracy, 'test_f1': f1, 'test_precision': precision, 'test_recall': recall, 'test_auc': roc_auc})
        
    metrics = ['test_acc', 'test_f1', 'test_precision', 'test_auc', 'test_recall']
    with open(output_metrics_raw, "w") as fin:
        for n, result in enumerate(fold_results):
            for metric in metrics:
                print(f"Baseline_{n}\t{rename[metric]}\t{result[metric]}", file=fin)
        
    calculate_means_and_cis_to_file(fold_results, output_metrics)

if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
