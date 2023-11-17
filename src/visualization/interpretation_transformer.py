import sys
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
    dataset = dataset_root / "dataset_v01"
    outf = Path("data/visualization/captum/integrated_gradients/inception/")
    outf.mkdir(parents=True, exist_ok=True)
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
    df = pd.read_csv(dataset / "test.tsv", sep="\t", header=0, names=["id", "type", "label"])
    test_labels = df["label"].values.tolist()
    # Next get coordinates..

    for n, (seq, label) in enumerate(data_module.test_dataloader()):

        out = model(seq)
        outs = softmax(out, dim=1)
        prob, l = topk(outs, 1)
        print(out, prob.detach().squeeze(), label.squeeze(), l.squeeze())
        
        integrated_gradients = IntegratedGradients(model)
        
        # attributions_ig = integrated_gradients.attribute(seq,
        #                                                         target=label,
        #                                                         n_steps=200)
        # attributions =  attributions_ig.detach().cpu().squeeze().numpy()
        # aggregated_attributions = np.sum(np.abs(attributions), axis=0)


        noise_tunnel = NoiseTunnel(integrated_gradients)

        attributions_nt = noise_tunnel.attribute(seq,target=label, nt_samples=10, nt_type='smoothgrad_sq')
        attributions =  attributions_nt.detach().cpu().squeeze().numpy()
        aggregated_attributions = np.sum(np.abs(attributions), axis=0)
        
        # Normalizing the aggregated attributions
        normalized_attributions = aggregated_attributions / np.linalg.norm(aggregated_attributions, ord=1)
        normalized_attributions = bin_attributions(normalized_attributions)
        # Visualizing the normalized attributions
        plt.figure(figsize=(15, 5))
        plt.plot(normalized_attributions, label='Importance')
        plt.xlabel('Sequence Position')
        plt.ylabel('Normalized Importance')
        plt.title('Aggregated Sequence Position Importance')
        plt.legend()
        plt.savefig(outf / "aggregated_importance_bins.png")
        plt.clf()
        sys.exit()


if __name__ == "__main__":
    main()

