import sys
import numpy as np
from torch import load as tload
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score, recall_score
from sklearn.linear_model import LogisticRegression
from src.models.LNSequenceModule import SequenceModule

def encode_sequence(sequence, vocabulary_mapping):
    return [vocabulary_mapping[gene_name] for gene_name in sequence]

def count_encode(samples, vocab_size):
    # Initialize a matrix of zeros with the shape (number of samples, vocabulary size)
    features = np.zeros((len(samples), vocab_size), dtype=int)

    for i, sample in enumerate(samples):
        for gene_code in sample:
            if not gene_code < vocab_size:
                print("ERROR: ", gene_code)
            else:
                features[i, gene_code] += 1
    
    return features

def main():

    np.random.seed(1)

    # Input
    dataset_root = Path("data/processed/10_datasets/")
    version = "dataset_v02"
    dataset = dataset_root / version
    checkpoint = Path("models/transformer/alldb_v02_small_iak7l6eg.ckpt")
    # Output
    output_root = Path("data/visualization/performance/linear/")
    outputfn = output_root / "logistic_alldb_v02.txt"

    model = SequenceModule.load_from_checkpoint(checkpoint_path=checkpoint, map_location="cpu")
    vocab_map = model.vocab_map

    data_splits = tload(dataset / "strings" / "allDB" / "dataset.pt", map_location="cpu")
    print("ORIGINAL: ")
    print(data_splits['train']['sequences'][0][:10])
    X_train = [encode_sequence(string, vocab_map) for string in data_splits['train']['sequences']]
    print("INTEGER ENCODING: ")
    print(X_train[0][:10])
    X_train = count_encode(X_train, len(vocab_map))
    print("COUNT ENCODING SHAPE: ")
    print(X_train.shape)
    y_train = data_splits['train']['labels'].numpy()

    model = LogisticRegression(penalty='l1', max_iter=1000, solver='saga', random_state=1)
    model.fit(X_train, y_train)

    X_test = [encode_sequence(string, vocab_map) for string in data_splits['test']['sequences']]
    X_test = count_encode(X_test, len(vocab_map))
    y_test = data_splits['test']['labels'].numpy()

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print(f'ROC-AUC Score: {roc_auc}')


    # for model_string, label in zip(data_splits['test']['sequences'], data_splits['test']['labels']):
        
    #     # Get sequence of HMM profile matches and pad
    #     encoded_string = torch.tensor(encode_sequence(model_string, vocab_map))
    #     padding = pad_len - len(encoded_string)
    #     encoded_string = F.pad(encoded_string, (0, padding), "constant", 0)
    #     sequence = encoded_string.unsqueeze(0)
    #     prediction = model(sequence)
    #     predictions.append(prediction)
    #     y_true.append(label.item())
    #     y_pred.append(prediction.argmax(dim=1).item())
    #     y_prob.append(softmax(prediction, dim=1)[0][1].item())

    # # calculate metrics
    # acc = accuracy_score(y_true, y_pred)
    # f1 = f1_score(y_true, y_pred, average='binary')
    # precision = precision_score(y_true, y_pred, average='binary')
    # auc = roc_auc_score(y_true, y_prob)


    # print("Accuracy: ", acc)
    # print("F1: ", f1)
    # print("Precision: ", precision)
    # print("AUC: ", auc)

    # with open(outputfn, "w") as fout:
    #     for tru, pred, prob, logits in zip(y_true, y_pred, y_prob, predictions):
    #         print(f"{tru}\t{pred}\t{prob}\t{logits.squeeze()[0].item()}\t{logits.squeeze()[1].item()}", file=fout)

if __name__ == "__main__":
    main()
