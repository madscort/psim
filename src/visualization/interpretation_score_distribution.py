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
    dataset = dataset_root / "dataset_v01"
    tax = Path("data/processed/03_taxonomical_annotation/ps_tax_info.tsv")
    tax_df = pd.read_csv(tax, sep="\t", header=0, names=["id", "family", "genus", "species"])
    df = pd.read_csv(dataset / "test.tsv", sep="\t", header=0, names=["id", "type", "label"])
    
    outf = Path("data/visualization/score_distribution/taxspecies/inception/")
    outf.mkdir(parents=True, exist_ok=True)
    outfn = outf / "ko59yma9.tsv"
    modelin = Path("models/inception/checkpoint/ko59yma9.ckpt")
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

    #type_labels = df["type"].values.tolist()

    # taxfamily = tax_df["family"].values.tolist()
    # tax_count = Counter(taxfamily)

    taxspecies = tax_df["species"].values.tolist()
    tax_count = Counter(taxspecies)

    common_tax = tax_count.most_common(20)
    toptax = [x[0] for x in common_tax]
    print(toptax)
    with open(outfn, "w") as fout:
        for n, (seq, label) in enumerate(data_module.test_dataloader()):
            if df['label'].values[n] == 0:
                continue
            # pred_id = type_labels[n]
            # try:
            #     pred_id = pred_id.split("_")[0]
            # except IndexError:
            #     pass

            # Get the taxonomic family of the sequence.
            # try:
            #     pred_id = tax_df[tax_df["id"] == df["id"].values[n]]["family"].values[0]
            # except IndexError:
            #     print(f"IndexError: {df['id'].values[n]}")
            #     continue

            try:
                pred_id = tax_df[tax_df["id"] == df["id"].values[n]]["species"].values[0]
            except IndexError:
                print(f"IndexError: {df['id'].values[n]}")
                continue
            
            # if pred_tax in toptax:
            #     pred_id = pred_tax
            # else:
            #     pred_id = "other"

            out = model(seq)
            outs = softmax(out, dim=1)
            prob = float(outs.squeeze()[1].item())
            print(pred_id,
                  f"{prob:.4f}",
                  int(label.squeeze()),
                  sep="\t",
                  file=fout)

if __name__ == "__main__":
    main()

