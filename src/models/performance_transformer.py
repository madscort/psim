import sys
import torch
from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything

from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score

from src.data.LN_data_module import FixedLengthSequenceModule
from src.models.LNSequenceModule import SequenceModule

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

    data_module = FixedLengthSequenceModule(dataset=dataset,
                                            return_type="hmm_match_sequence",
                                            num_workers=1,
                                            batch_size=2,
                                            pad_pack=True,
                                            use_saved=True)
    data_module.setup()
    model = SequenceModule.load_from_checkpoint(checkpoint_path=checkpoint, map_location="cpu")
    model.eval()
    model.to("cpu")
    trainer = Trainer(accelerator="cpu")

    trainer.test(model, datamodule=data_module)

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
    y_hat = y_hat.cpu()
    
    # Write tsv with: 
    # with open(outputfn, "w") as fout:
    #     for i in range(len(y_true)):
    #         fout.write(f"{y_true[i]}\t{y_pred_prob_class_1[i]}\t{y_pred[i]}\n")

    # calculate metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary')
    precision = precision_score(y_true, y_pred, average='binary')
    auc = roc_auc_score(y_true, y_pred_prob_class_1)

    # log ROC curve in wandb
    y_true_np = y_true.numpy()
    y_hat_np = y_pred_prob.numpy()

    print("Accuracy: ", acc)
    print("F1: ", f1)
    print("Precision: ", precision)
    print("AUC: ", auc)


if __name__ == "__main__":
    main()
