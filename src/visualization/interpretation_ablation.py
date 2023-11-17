import sys
import numpy as np
from pathlib import Path
from pytorch_lightning import seed_everything
from torch.nn.functional import softmax
from torch import topk
from captum.attr import FeatureAblation
import matplotlib.pyplot as plt
from src.data.LN_data_module import FixedLengthSequenceModule
from src.models.LNSequenceModule import SequenceModule

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

    for seq, label in data_module.test_dataloader():
        if label == 0:
            print("Skipping")
            continue
        out = model(seq)
        outs = softmax(out, dim=1)
        prob, l = topk(outs, 1)
        print(out, prob.detach().squeeze(), label.squeeze(), l.squeeze())
        
        ablator = FeatureAblation(model)

        attributions = ablator.attribute(seq, target=label.squeeze(), show_progress=True)
        
        # Processing attributions for visualization
        attributions = attributions.detach().cpu().squeeze().numpy()
        aggregated_attributions = np.mean(np.abs(attributions), axis=0)

        # Visualizing the aggregated attributions
        plt.figure(figsize=(15, 5))
        plt.bar(range(len(aggregated_attributions)), aggregated_attributions)
        plt.xlabel('Sequence Position')
        plt.ylabel('Importance')
        plt.title('Nucleotide Position Importance via Feature Ablation')
        plt.savefig(outf / "feature_ablation_importance.png")
        plt.clf()
        sys.exit()

if __name__ == "__main__":
    main()

