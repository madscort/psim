import hydra
import sys
from collections import Counter
import mappy as mp
from tempfile import TemporaryDirectory

from pathlib import Path
from omegaconf import DictConfig
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import pandas as pd

def mapq2prob(quality_score):
    return 10 ** (-quality_score / 10)


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
                for sequence_path in data_splits['train']['sequences']:
                    with open(sequence_path, 'r') as g:
                        f.write(g.read())

            # Load as index:
            a = mp.Aligner(str(train_fasta))  # load or build index
            if not a:
                raise Exception("ERROR: failed to load/build index")
            
            # Align each test sequence to the index
            max_scores = []
            max_labels = []
            label_predict = []
            for label, sequence_path in zip(data_splits['test']['labels'],data_splits['test']['sequences']):
                max_score = 0
                match_id = None
                for name, seq, qual in mp.fastx_read(str(sequence_path)):
                    # align sequence to index
                    for hit in a.map(seq): # traverse alignments
                        if hit.mapq > max_score:
                            max_score = hit.mapq
                            match_id = hit.ctg
                if match_id is not None:
                    label_predict.append(df_sampletable.loc[df_sampletable['id'] == match_id]['label'].values[0])
                else:
                    label_predict.append(0)
                max_scores.append(1 - mapq2prob(max_score))
                max_labels.append(label)
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
        plt.show()

        # y_hat = outputs[0]['y_hat'].cpu().numpy()
        # y = outputs[0]['y'].cpu().numpy()

        # # Save predictions and ground truth labels for this fold
        # fold_results.append((y_hat, y))

    # Save the results to disk
    # with open(Path('data/visualization/cross_validation/cross_val_results.pkl').absolute(), 'wb') as f:
    #     pickle.dump(fold_results, f)

if __name__ == "__main__":
    main()