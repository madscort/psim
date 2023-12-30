import sys
import numpy as np
import pandas as pd
from pathlib import Path
from pytorch_lightning import seed_everything
from torch.nn.functional import softmax
from torch import norm, full_like, empty
from torch import long as tlong
from captum.attr import LayerIntegratedGradients
import matplotlib.pyplot as plt
from src.data.LN_data_module import FixedLengthSequenceModule
from src.models.LNSequenceModule import SequenceModule

def main():

    seed_everything(1)
    
    dataset_root = Path("data/processed/10_datasets/")
    dataset = dataset_root / "dataset_v02"
    outf = Path("data/visualization/captum/integrated_gradients/transformer/")
    outf.mkdir(parents=True, exist_ok=True)
    outfn = outf / "aggregated_lig_alldb_prophage_baseline_v02.tsv"
    modelin = Path("models/transformer/alldb_v02_small_iak7l6eg.ckpt")
    data_module = FixedLengthSequenceModule(dataset=dataset,
                                            return_type="hmm_match_sequence",
                                            num_workers=1,
                                            batch_size=1,
                                            pad_pack=True,
                                            use_saved=True)
    data_module.setup()

    model = SequenceModule.load_from_checkpoint(checkpoint_path=modelin,
                                                map_location="cpu")
    vocab = model.vocab_map
    inv_vocab = {v: k for k, v in vocab.items()}

    model.eval()
    
    df = pd.read_csv(dataset / "test.tsv", sep="\t", header=0, names=["id", "type", "label"])
    test_labels = df["label"].values.tolist()
    test_types = df["type"].values.tolist()
    
    with open(outfn, "w") as fout:
        for n, (seq, label) in enumerate(data_module.test_dataloader()):
            if not test_types[n].startswith("provirus"):
                continue
            # if label == 0:
            #     continue
            proteins = [inv_vocab[x] for x in seq.squeeze().detach().numpy()]
            baseline = full_like(empty(1, 56,dtype=tlong), vocab["no_hit"])
            lig = LayerIntegratedGradients(model, model.model.embedding)
            attributions_ig = lig.attribute(inputs=seq, baselines=baseline, target=0, n_steps=200)
            attributions = attributions_ig.sum(dim=-1).squeeze(0)
            attributions = attributions / norm(attributions)

            for protein, attribution in zip(proteins, attributions):
                origin = "unknown"
                if protein.startswith("IMGVR"):
                    origin = "provirus"
                elif protein.startswith("PS"):
                    origin = "satellite"
                elif protein.startswith("N") or protein.startswith("D") or protein.startswith("C"):
                    origin = "host"
                elif protein.startswith("S"):
                    origin = "metagenome"

                print(protein,
                    attribution.item(),
                    origin,
                    sep="\t", file=fout)

            # Visualizing the normalized attributions
            plt.figure(figsize=(20, 5))
            plt.barh(proteins, attributions)
            plt.ylabel("Proteins")
            plt.xlabel("Attribution")
            plt.title("Integrated Gradients Attributions per Protein")
            plt.xticks(rotation=45)
            plt.show()


if __name__ == "__main__":
    main()

