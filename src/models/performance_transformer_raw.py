import sys
import torch
from pathlib import Path
from pytorch_lightning import seed_everything
from torch.nn.functional import softmax

from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score

from src.models.LNSequenceModule import SequenceModule

def encode_sequence(sequence, vocabulary_mapping):
    return [vocabulary_mapping[gene_name] for gene_name in sequence]

def main():

    seed_everything(1)
    
    # Input
    dataset_root = Path("data/processed/10_datasets/")
    version = "dataset_v01"
    dataset = dataset_root / version
    checkpoint = Path("models/transformer/alldb_small_wodict_ne6zbqji.ckpt")

    # Output
    output_root = Path("data/visualization/performance")
    outputfn = output_root / "alldb_small_wodict_ne6zbqji.txt"

    model = SequenceModule.load_from_checkpoint(checkpoint_path=checkpoint, map_location="cpu")
    model.to("cpu")
    model.eval()

    vocab_map = model.vocab_map

    data_splits = torch.load(dataset / "strings" / "allDB" / "dataset.pt", map_location="cpu")
    y_true = []
    y_pred = []
    y_prob = []
    
    for model_string, label in zip(data_splits['test']['sequences'], data_splits['test']['labels']):
        encoded_string = torch.tensor(encode_sequence(model_string, vocab_map))
        sequence = encoded_string.unsqueeze(0)
        print(sequence)
        sys.exit()
        if len(sequence) < 56:
            torch.cat([sequence[0], torch.zeros(56 - len(sequence))])
        prediction = model(sequence)
        y_true.append(label.item())
        y_pred.append(prediction.argmax(dim=1).item())
        y_prob.append(softmax(prediction, dim=1)[0][1].item())

    print(len(y_true))
    
    # Write tsv with: 
    # with open(outputfn, "w") as fout:
    #     for i in range(len(y_true)):
    #         fout.write(f"{y_true[i]}\t{y_pred_prob_class_1[i]}\t{y_pred[i]}\n")

    # calculate metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary')
    precision = precision_score(y_true, y_pred, average='binary')
    auc = roc_auc_score(y_true, y_prob)


    print("Accuracy: ", acc)
    print("F1: ", f1)
    print("Precision: ", precision)
    print("AUC: ", auc)


if __name__ == "__main__":
    main()
