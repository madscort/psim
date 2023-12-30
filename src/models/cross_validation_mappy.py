import hydra
import sys
from collections import Counter
import mappy as mp
from tempfile import TemporaryDirectory

from pathlib import Path
from omegaconf import DictConfig
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from scipy.stats import t

def mapq2prob(quality_score):
    return 10 ** (-quality_score / 10)

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

def load_split_data(dataset, split):
    df = pd.read_csv(dataset / f"{split}.tsv", sep="\t", header=0, names=["id", "type", "label"])
    sequences = [dataset / split / "sequences" / f"{id}.fna" for id in df['id'].values]
    labels = df['label'].values
    return {'sequences': sequences, 'labels': labels}

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

@hydra.main(config_path="../../configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig):

    output_metrics = Path("data/visualization/performance/v02/confidence/cross_mappy.tsv")
    output_metrics_raw = Path("data/visualization/performance/v02/confidence/cross_mappy_raw.tsv")
    dataset_root = Path("data/processed/10_datasets/")
    dataset = Path(dataset_root, cfg.dataset)
    sampletables = [pd.read_csv(dataset / f"{split}.tsv", sep="\t", header=0, names=["id", "type", "label"]) for split in ['train', 'val', 'test']]
    df_sampletable = pd.concat(sampletables, ignore_index=True)

    data_splits = {
                split: load_split_data(dataset, split) for split in ['train', 'val', 'test']
            }

    # Initialize the StratifiedKFold object
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    fold_results = []

    for fold, (fit_idx, test_idx) in enumerate(skf.split(df_sampletable['id'], df_sampletable['type'])):
        print(f"Fold {fold}")
        # Split the training data into training and validation sets

        train_idx, val_idx = custom_stratify(df_sampletable.iloc[fit_idx], 'type')
        
        data_splits = resplit_data(data_splits=data_splits, train_indices=train_idx, val_indices=val_idx, test_indices=test_idx)

        with TemporaryDirectory() as tmp:
            # Create fasta for training data
            train_fasta = Path(tmp) / "train.fna"
            with open(train_fasta, "w") as f:
                # Only positive labels
                for label, sequence_path in zip(data_splits['train']['labels'],data_splits['train']['sequences']):
                    if label == 0:
                        continue
                    with open(sequence_path, 'r') as g:
                        f.write(g.read())

            # Load as index:
            a = mp.Aligner(str(train_fasta))  # load or build index
            if not a:
                raise Exception("ERROR: failed to load/build index")
            
            # Align each test sequence to the index
            y_prob = []
            y_true = []
            y_pred = []
            for label, sequence_path in zip(data_splits['test']['labels'],data_splits['test']['sequences']):
                max_score = 0
                match_id = None
                for name, seq, qual in mp.fastx_read(str(sequence_path)):
                    
                    # align sequence to index
                    for hit in a.map(seq):
                        if hit.mapq > max_score:
                            max_score = hit.mapq
                            match_id = hit.ctg
                if match_id is not None:
                    y_pred.append(df_sampletable.loc[df_sampletable['id'] == match_id]['label'].values[0])
                else:
                    y_pred.append(0)
                y_prob.append(1 - mapq2prob(max_score))
                y_true.append(label)

        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='binary')
        precision = precision_score(y_true, y_pred, average='binary')
        recall = recall_score(y_true, y_pred, average='binary')
        roc_auc = roc_auc_score(y_true, y_prob)
        print(f"Accuracy: {accuracy}")
        print(f"F1: {f1}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"ROC AUC: {roc_auc}")

        fold_results.append({'test_acc': accuracy, 'test_f1': f1, 'test_precision': precision, 'test_recall': recall, 'test_auc': roc_auc})

    metrics = ['test_acc', 'test_f1', 'test_precision', 'test_auc', 'test_recall']
    with open(output_metrics_raw, "w") as fin:
        for n, result in enumerate(fold_results):
            for metric in metrics:
                print(f"Baseline_{n}\t{rename[metric]}\t{result[metric]}", file=fin)
        
    calculate_means_and_cis_to_file(fold_results, output_metrics)


if __name__ == "__main__":
    main()