import hydra
import wandb
import sys
from pathlib import Path
from omegaconf import DictConfig
from omegaconf.omegaconf import OmegaConf
import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelSummary

from src.data.LN_data_module import FixedLengthSequenceModule
from src.models.CNN_captum import CaptumModule

import captum
from captum.attr import IntegratedGradients, Occlusion, LayerGradCam, LayerAttribution
from captum.attr import visualization as viz
from captum.attr import Occlusion

seed_everything(10)

dataset_root = Path("data/processed/10_datasets/")
dataset = Path(dataset_root, "phage_25_fixed_25000_reduced_90")
data_module = FixedLengthSequenceModule(dataset=dataset,
                                        return_type="fna",
                                        num_workers=0,
                                        batch_size=1,
                                        pad_pack=False,
                                        use_saved=False)

data_module.setup()
model = CaptumModule().load_from_checkpoint(Path("psim_captum/bx3s8970/checkpoints/epoch=23-step=4104.ckpt").absolute())
model.eval()
model.to('cpu')
train_dataloader = data_module.train_dataloader()
seqs, labels = next(iter(train_dataloader))

occlusion = Occlusion(model)

print(seqs.shape)
print(labels)


output = model(seqs)
output = output.detach()
output = F.softmax(output, dim=1)
prediction_score, pred_label_idx = torch.topk(output, 1)
print(pred_label_idx)
# Set up hyperparameters
strides = (5,10)
sliding_window_shapes = (5,100)

attributions = occlusion.attribute(seqs,
                                    sliding_window_shapes=sliding_window_shapes,
                                    strides=strides,
                                    target=pred_label_idx)
import matplotlib.pyplot as plt
import numpy as np

attributions_numpy = attributions.squeeze().cpu().detach().numpy()

for i in range(5):  # Assuming 5 channels
    plt.plot(attributions_numpy[:, i], label=f'Channel {i+1}')

plt.xlabel('Sequence Position')
plt.ylabel('Attribution Value')
plt.title('Occlusion-based Attributions')
plt.legend()

plt.savefig(f"data/visualization/test/occlusion_epoch_label{labels[0].item()}.png")

# import seaborn as sns

# attributions_numpy = attributions.squeeze().cpu().detach().numpy()

# sns.heatmap(attributions_numpy, annot=True, fmt=".2g")
# plt.xlabel('Channel')
# plt.ylabel('Sequence Position')
# plt.title('Occlusion-based Attributions')
# plt.show()


sys.exit()
for num, batch in enumerate(train_dataloader):
    seqs, label = batch

    # [0] is sample 1
    output = model(seqs)
    output = output.detach()
    # print("model output: ", output)
    output = F.softmax(output, dim=1)
    # print("softmax on output: ", output)
    prediction_score, pred_label_idx = torch.topk(output, 1)
    # print("Score: ", prediction_score, "Label: ", pred_label_idx)
    # print("Squeeze dimensions: ", pred_label_idx.squeeze())

    # Initialize the attribution algorithm with the model
    integrated_gradients = IntegratedGradients(model)

    # Ask the algorithm to attribute our output target to
    attributions_ig = integrated_gradients.attribute(seqs, target=pred_label_idx)

    # Assuming attributions_ig is a PyTorch tensor with shape [num_samples, num_channels, sequence_length]
    # Converting it to numpy for visualization
    attributions = attributions_ig.detach().cpu().numpy()

    # If you want to visualize just the first sample, make it [num_channels, sequence_length]
    attributions = attributions[0].squeeze()

    num_channels = attributions.shape[0]

    fig, axes = plt.subplots(nrows=num_channels, figsize=(15, 20), sharex=True)

    # Create a shared colorbar
    cax = fig.add_axes([0.92, 0.11, 0.02, 0.77])

    for i in range(num_channels):
        im = axes[i].imshow(attributions[i][np.newaxis,:], cmap="viridis", aspect="auto")
        axes[i].set_title(f"Channel {i+1}")

    # Create a single colorbar for all subplots
    fig.colorbar(im, cax=cax, orientation='vertical')

    plt.savefig(f"data/visualization/test/captum_{num}_epoch_label{label[0].item()}.png")
    plt.clf()
