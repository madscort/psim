import numpy as np
import logging
from torch import load as tload
from pathlib import Path
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
    output_root = root / "data/visualization/performance/v02/"
    output_root.mkdir(parents=True, exist_ok=True)
    outputfn = output_root / "logistic_alldb_c1_v02_performance.txt"
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

    c_value = 1
    
    model = LogisticRegression(penalty='l1', solver='saga', C=c_value, max_iter=5000, random_state=1, n_jobs=6)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    
    with open(outputfn, 'w') as fout:
        for n in range(len(y_val)):
            print(y_val[n], y_pred[n], y_prob[n], sep='\t', file=fout)

if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
