import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from pytorch_lightning import seed_everything
from torch.nn.functional import softmax
from torch import topk
from captum.attr import IntegratedGradients
from captum.attr import NoiseTunnel
import matplotlib.pyplot as plt
from src.data.LN_data_module import FixedLengthSequenceModule
from src.models.LNSequenceModule import SequenceModule

def bin_attributions(attributions, bin_size=3):
    bin_size = min(bin_size, len(attributions))

    binned_attributions = [
        np.mean(attributions[i:i+bin_size]) 
        for i in range(0, len(attributions) - bin_size + 1, bin_size)
    ]
    return binned_attributions


def main():

    seed_everything(1)
    
    dataset_root = Path("data/processed/10_datasets/")
    dataset = dataset_root / "dataset_v02"
    outf = Path("data/visualization/captum/integrated_gradients/inception/aggregated_importance")
    outf.mkdir(parents=True, exist_ok=True)
    outf_plot = outf / "plots"
    outf_plot.mkdir(parents=True, exist_ok=True)
    outf_raw = outf / "raw"
    outf_raw.mkdir(parents=True, exist_ok=True)
    modelin = Path("models/inception/checkpoint/24eamlea_version02.ckpt")
    data_module = FixedLengthSequenceModule(dataset=dataset,
                                            return_type="fna",
                                            num_workers=1,
                                            batch_size=1,
                                            pad_pack=False,
                                            use_saved=False)
    data_module.setup()
    model = SequenceModule.load_from_checkpoint(checkpoint_path=modelin, map_location='cpu')

    model.eval()
    model.to('cpu')
    df = pd.read_csv(dataset / "test.tsv", sep="\t", header=0, names=["id", "type", "label"])
    df_coord = pd.read_csv(dataset / "test" / "contig_coordinates.tsv", sep="\t", header=None, names=["id", "start", "end"])
    df_taxonomy = pd.read_csv("data/processed/03_taxonomical_annotation/ps_tax_info.tsv", sep="\t", header=0, names=["id", "family", "genus", "species"])
    test_ids = df["id"].values.tolist()
    
    for n, (seq, label) in enumerate(data_module.test_dataloader()):
        if label == 0:
            continue
        id = test_ids[n]
        start = df_coord[df_coord["id"] == test_ids[n]]["start"].values[0]
        end = df_coord[df_coord["id"] == test_ids[n]]["end"].values[0]
        species = df_taxonomy[df_taxonomy["id"] == test_ids[n]]["species"].values[0]
        genus = df_taxonomy[df_taxonomy["id"] == test_ids[n]]["genus"].values[0]
        print("ID: ", test_ids[n])
        print("Label: ", label)
        print("Start: ", start)
        print("End: ", end)
        print("Species: ", species)
        print("Genus: ", genus)
        out = model(seq)
        outs = softmax(out, dim=1)
        prob, _ = topk(outs, 1)
        prob = prob.item()

        integrated_gradients = IntegratedGradients(model)
        
        # Raw integrated gradients
        # attributions_ig = integrated_gradients.attribute(seq,
        #                                                         target=label,
        #                                                         n_steps=200)
        # attributions =  attributions_ig.detach().cpu().squeeze().numpy()
        # aggregated_attributions = np.sum(np.abs(attributions), axis=0)

        noise_tunnel = NoiseTunnel(integrated_gradients)
        attributions_nt = noise_tunnel.attribute(seq,target=label, nt_samples=5, nt_type='smoothgrad_sq')
        attributions =  attributions_nt.detach().cpu().squeeze().numpy()
        aggregated_attributions = np.sum(np.abs(attributions), axis=0)
        
        # Optional normalization:
        #aggregated_attributions = aggregated_attributions / np.linalg.norm(aggregated_attributions, ord=1)
        
        # Optional binning:
        #aggregated_attributions = bin_attributions(aggregated_attributions)
    
        # Visualizing
        plt.figure(figsize=(15, 5))
        plt.plot(aggregated_attributions, label='Importance')
        plt.hlines(y=0, xmin=start, xmax=end, color='red', linewidth=5)
        plt.xlabel('Sequence Position')
        plt.ylabel('Importance')
        plt.title('Aggregated Sequence Position Importance')
        plt.legend()
        plt.savefig(outf_plot / f"{id}.png")
        plt.clf()
        # Save raw data:
        with open(outf_raw / f"{id}.tsv", "w") as fout:
            for pos, imp in enumerate(aggregated_attributions):
                print(id,
                        species,
                        genus,
                        start,
                        end,
                        pos,
                        imp,
                        prob,
                        sep="\t",
                        file=fout)


if __name__ == "__main__":
    main()

