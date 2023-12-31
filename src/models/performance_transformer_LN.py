import sys
import torch
from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything

from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score, recall_score, confusion_matrix

from src.data.LN_data_module import FixedLengthSequenceModule
from src.models.LNSequenceModule import SequenceModule

def main():

    seed_everything(1)
    
    # Input
    dataset_root = Path("data/processed/10_datasets/")
    version = "dataset_v02"
    dataset = dataset_root / version

    checkpoint = Path("models/transformer/mmdb_single_v02_small_ofkn1eok.ckpt")

    # Output
    output_root = Path("data/visualization/performance/v02/validation")
    output_root.mkdir(parents=True, exist_ok=True)
    outputfn = output_root / "mmdb_single_v02_small_ofkn1eok_performance.tsv"

    data_module = FixedLengthSequenceModule(dataset=dataset,
                                            return_type="hmm_match_sequence",
                                            num_workers=4,
                                            batch_size=32,
                                            pad_pack=True,
                                            use_saved=True)
    data_module.setup()
    model = SequenceModule.load_from_checkpoint(checkpoint_path=checkpoint, map_location="cpu")
    model.eval()
    trainer = Trainer(accelerator="cpu")

    trainer.validate(model, datamodule=data_module)

    y_true = torch.cat([x['y_true'] for x in trainer.results], dim=0)
    y_hat = torch.cat([x['y_hat'] for x in trainer.results], dim=0)
    
    print(len(y_true))

    # Convert probabilities to predicted labels
    y_pred_prob = torch.nn.functional.softmax(y_hat, dim=1)
    y_pred = y_pred_prob.argmax(dim=1)

    y_true = y_true.cpu()
    y_pred_prob = y_pred_prob.cpu()
    y_pred_prob_class_1 = y_pred_prob[:, 1] 
    y_pred = y_pred.cpu()
    #y_hat = y_hat.cpu()

    # calculate metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary')
    precision = precision_score(y_true, y_pred, average='binary')
    auc = roc_auc_score(y_true, y_pred_prob_class_1)
    recall = recall_score(y_true, y_pred, average='binary')
    cm = confusion_matrix(y_true, y_pred)

    print("Accuracy: ", acc)
    print("F1: ", f1)
    print("Precision: ", precision)
    print("AUC: ", auc)
    print("Recall: ", recall)
    print("Confusion Matrix:\n", cm)

    # # Write tsv with: 
    with open(outputfn, "w") as fout:
        for tru, pred, prob, logits in zip(y_true, y_pred, y_pred_prob_class_1, y_hat):
            print(f"{tru}\t{pred}\t{prob}\t{logits.squeeze()[0].item()}\t{logits.squeeze()[1].item()}", file=fout)

if __name__ == "__main__":
    main()
