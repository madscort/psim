import sys
import numpy as np
import pandas as pd
from pathlib import Path
from pytorch_lightning import seed_everything
from torch.nn.functional import softmax
from src.data.LN_data_module import FixedLengthSequenceModule
from src.models.LNSequenceModule import SequenceModule
from collections import Counter


def main():

    seed_everything(1)
    
    dataset_root = Path("data/processed/10_datasets/")
    dataset = dataset_root / "dataset_v02"
    tax = Path("data/processed/03_taxonomical_annotation/ps_tax_info.tsv")
    tax_df = pd.read_csv(tax, sep="\t", header=0, names=["id", "family", "genus", "species"])
    df = pd.read_csv(dataset / "test.tsv", sep="\t", header=0, names=["id", "type", "label"])
    
    outf = Path("data/visualization/score_distribution/v02/genus_type/inception/")
    outf.mkdir(parents=True, exist_ok=True)
    outfn = outf / "24eamlea_version02.tsv"
    modelin = Path("models/inception/checkpoint/24eamlea_version02.ckpt")
    data_module = FixedLengthSequenceModule(dataset=dataset,
                                            return_type="fna",
                                            num_workers=1,
                                            batch_size=1,
                                            pad_pack=False,
                                            use_saved=False)
    data_module.setup()

    model = SequenceModule.load_from_checkpoint(checkpoint_path=modelin,
                                                map_location="cpu")
    model.eval()
    model.to('cpu')
    type_labels = df["type"].values.tolist()

    # taxfamily = tax_df["family"].values.tolist()
    # tax_count = Counter(taxfamily)

    taxgenus = tax_df["genus"].values.tolist()
    tax_count = Counter(taxgenus)

    # taxspecies = tax_df["species"].values.tolist()
    # tax_count = Counter(taxspecies)

    # common_tax = tax_count.most_common(20)
    # toptax = [x[0] for x in common_tax]
    # print(toptax)
    with open(outfn, "w") as fout:
        for n, (seq, label) in enumerate(data_module.test_dataloader()):
            
            # Just label:
            #pred_id = int(label.squeeze())
            
            if df['label'].values[n] == 0:
                continue
            
            # pred_id = type_labels[n]
            
            # # Rough label type
            # pred_id = type_labels[n]
            # if int(label.squeeze()) == 0:
            #     if pred_id.startswith("pro"):
            #         pred_id = "Prophage"
            #     elif pred_id.startswith("meta"):
            #         pred_id = "Metagenomic"
            #     elif pred_id.startswith("host"):
            #         pred_id = "Host"

            # Get origin of sequence
            # type_id = pred_id
            # if type_id.startswith("prov"):
            #     pred_id = "provirus"
            # elif type_id.startswith("host"):
            #     pred_id = "host"
            # elif type_id.startswith("meta"):
            #     pred_id = "metagenome"
            # else:
            #     pred_id = "satellite"

            # Get the taxonomic family of the sequence.
            # try:
            #     pred_id = tax_df[tax_df["id"] == df["id"].values[n]]["family"].values[0]
            # except IndexError:
            #     print(f"IndexError: {df['id'].values[n]}")
            #     continue

            try:
                pred_id = tax_df[tax_df["id"] == df["id"].values[n]]["genus"].values[0]
            except IndexError:
                print(f"IndexError: {df['id'].values[n]}")
                continue

            # try:
            #     pred_id = tax_df[tax_df["id"] == df["id"].values[n]]["species"].values[0]
            # except IndexError:
            #     print(f"IndexError: {df['id'].values[n]}")
            #     continue
            
            # if pred_tax in toptax:
            #     pred_id = pred_tax
            # else:
            #     pred_id = "other"

            out = model(seq)
            outs = softmax(out, dim=1)
            prob = float(outs.squeeze()[1].item())
            print(pred_id,
                  type_labels[n],
                  f"{prob:.4f}",
                  int(label.squeeze()),
                  sep="\t",
                  file=fout)

if __name__ == "__main__":
    main()

