from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from src.data.LN_data_module import FixedLengthSequenceModule
from src.models.CNN_basic import SequenceCNN
from sklearn.model_selection import train_test_split
import torch
import sys
import pandas as pd
import numpy as np
from src.models.CNN_collection import BasicCNN
import matplotlib.pyplot as plt


def check_dataloader_output(dataloader):
    for batch in dataloader:
        inputs, targets = batch
        print(f"Inputs: {inputs}")
        print(f"Targets: {targets}")
        # print(f"Inputs Shape: {torch.tensor(inputs).shape}")
        print(f"Inputs Shape: {inputs.shape}")
        print(f"Targets Shape: {targets.shape}")
        print(f"Sample Targets: {targets[:5]}")  # Print first 5 targets for inspection.
        # new = inputs.view(inputs.size(0), -1, 4)
        # print(f"New shape: {new.shape}")
        break # Break after the first batch.

data_module = FixedLengthSequenceModule(num_workers=0,
                                        return_type="fna",
                                        batch_size=1,
                                        dataset=Path("data/processed/10_datasets/phage_25_fixed_25000_reduced_90"),
                                        use_saved=False,
                                        pad_pack=False)
data_module.setup()

# Check train DataLoader
train_dataloader = data_module.train_dataloader()

print("Train DataLoader:")
check_dataloader_output(train_dataloader)

from src.models.LNSequenceModule import SequenceModule

model = SequenceModule(model = 'BasicCNN', ckpt_path= "psim/80ohj6m3/checkpoints/epoch=50-step=8721.ckpt")

model.eval()
# for batch in train_dataloader:
#     seqs, labels = batch
#     outputs = model(seqs)
#     outputs.backward()
#     gradients = model.get_activations_gradient()
#     print("Grad shape: ", gradients.shape)
#     pooled_gradients = torch.mean(gradients, dim=(0, 1))
#     print("Pooled gradients: ", pooled_gradients.shape)

#     activations = model.get_activations(seqs).detach()
#     heatmap = torch.mean(activations, dim=1).squeeze()
#     heatmap /= torch.max(heatmap)
#     heatmap = heatmap.detach().numpy()
#     plt.imshow(heatmap[np.newaxis,:], aspect='auto', cmap='hot', extent=[0, len(heatmap), 0, 1])
#     plt.xlabel('Sequence Position')
#     plt.yticks([])
#     plt.title('1D Heatmap')
#     plt.colorbar(label='Activation Intensity', orientation='vertical')
#     plt.savefig("data/visualization/testfig.png")
#     break

for batch in train_dataloader:
    seqs, labels = batch
    outputs = model(seqs)

    # Get the activations
    activations = model.get_activations(seqs).detach()
    
    # Loop over each class
    for i in range(2):
        # Zero gradients
        model.zero_grad()
        
        # Backward pass with respect to a particular class
        outputs[:, i].backward(torch.ones_like(outputs[:, i]), retain_graph=True)
        
        # Get the gradients and compute the CAM
        gradients = model.get_activations_gradient()
        heatmap = torch.mean(activations, dim=1).squeeze() * torch.mean(gradients, dim=(0, 1)).squeeze()
        
        # Normalize the heatmap
        heatmap /= torch.max(heatmap)
        
        # Visualize the heatmap
        plt.subplot(2, 1, i+1)
        plt.imshow(heatmap[np.newaxis, :], aspect='auto', cmap='hot', extent=[0, len(heatmap), 0, 1])
        plt.title(f'Class {i} Activation Map')
        plt.colorbar(label='Activation Intensity', orientation='vertical')
    

    plt.savefig("data/visualization/testfig_trained.png")
    break

sys.exit()

# Check val DataLoader (if defined in your DataModule)
if hasattr(data_module, 'val_dataloader') and callable(data_module.val_dataloader):
    val_dataloader = data_module.val_dataloader()
    print("\nValidation DataLoader:")
    check_dataloader_output(val_dataloader)

# # Check test DataLoader (if defined in your DataModule)
if hasattr(data_module, 'test_dataloader') and callable(data_module.test_dataloader):
    test_dataloader = data_module.test_dataloader()
    print("\nTest DataLoader:")
    check_dataloader_output(test_dataloader)
