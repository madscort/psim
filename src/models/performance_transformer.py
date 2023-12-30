import sys
import torch
from pathlib import Path
from pytorch_lightning import seed_everything
from torch.nn.functional import softmax
import torch.nn.functional as F
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
    checkpoint = Path("psim/06kbqyvr/checkpoints/epoch=29-step=8010.ckpt")

    # Output
    output_root = Path("data/visualization/performance")
    outputfn = output_root / "transformer_alldb_small_wodict_ne6zbqji.txt"

    model = SequenceModule.load_from_checkpoint(checkpoint_path=checkpoint, map_location="cpu")
    model.eval()
    model.to("cpu")

    vocab_map = model.vocab_map

    data_splits = torch.load(dataset / "strings" / "allDB" / "dataset.pt", map_location="cpu")
    predictions = []
    y_true = []
    y_pred = []
    y_prob = []
    pad_len = 56

    
    for model_string, label in zip(data_splits['test']['sequences'], data_splits['test']['labels']):
        
        # Get sequence of HMM profile matches and pad
        encoded_string = torch.tensor(encode_sequence(model_string, vocab_map))
        padding = pad_len - len(encoded_string)
        encoded_string = F.pad(encoded_string, (0, padding), "constant", 0)
        sequence = encoded_string.unsqueeze(0)
        prediction = model(sequence)
        predictions.append(prediction)
        y_true.append(label.item())
        y_pred.append(prediction.argmax(dim=1).item())
        y_prob.append(softmax(prediction, dim=1)[0][1].item())

    # calculate metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary')
    precision = precision_score(y_true, y_pred, average='binary')
    auc = roc_auc_score(y_true, y_prob)


    print("Accuracy: ", acc)
    print("F1: ", f1)
    print("Precision: ", precision)
    print("AUC: ", auc)

    with open(outputfn, "w") as fout:
        for tru, pred, prob, logits in zip(y_true, y_pred, y_prob, predictions):
            print(f"{tru}\t{pred}\t{prob}\t{logits.squeeze()[0].item()}\t{logits.squeeze()[1].item()}", file=fout)

if __name__ == "__main__":
    main()
