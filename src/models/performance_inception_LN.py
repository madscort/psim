import sys
import torch
from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything

from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score, recall_score

from src.data.LN_data_module import FixedLengthSequenceModule
from src.models.LNSequenceModule import SequenceModule

def main():

    seed_everything(1)
    
    # Input
    dataset_root = Path("data/processed/10_datasets/")
    version = "dataset_v02"
    dataset = dataset_root / version

    checkpoint = Path("models/inception/checkpoint/24eamlea_version02.ckpt")

    # Output
    output_root = Path("data/visualization/performance")
    outputfn = output_root / "inception_performance_v02.tsv"

    data_module = FixedLengthSequenceModule(dataset=dataset,
                                            return_type="fna",
                                            num_workers=4,
                                            batch_size=16,
                                            pad_pack=False,
                                            use_saved=False)
    data_module.setup()
    model = SequenceModule.load_from_checkpoint(checkpoint_path=checkpoint, map_location="cpu")
    model.eval()
    trainer = Trainer(accelerator="cpu")
    trainer.test(model, datamodule=data_module)

    y_true = torch.cat([x['y_true'] for x in trainer.results], dim=0)
    y_hat = torch.cat([x['y_hat'] for x in trainer.results], dim=0)
    
    print(len(y_true))

    # Convert probabilities to predicted labels
    y_pred_prob = torch.nn.functional.softmax(y_hat, dim=1)
    y_pred = y_pred_prob.argmax(dim=1)

    # Move to cpu
    y_true = y_true.cpu()
    y_pred_prob = y_pred_prob.cpu()
    y_pred_prob_class_1 = y_pred_prob[:, 1] 
    y_pred = y_pred.cpu()

    # calculate metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary')
    precision = precision_score(y_true, y_pred, average='binary')
    auc = roc_auc_score(y_true, y_pred_prob_class_1)
    recall = recall_score(y_true, y_pred)

    print("Accuracy: ", acc)
    print("F1: ", f1)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("AUC: ", auc)

    # Write tsv with: 
    with open(outputfn, "w") as fout:
        for tru, pred, prob, logits in zip(y_true, y_pred, y_pred_prob_class_1, y_hat):
            print(f"{tru}\t{pred}\t{prob}\t{logits.squeeze()[0].item()}\t{logits.squeeze()[1].item()}", file=fout)


if __name__ == "__main__":
    main()
