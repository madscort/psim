from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from src.data.LN_data_module import FixedLengthSequenceModule
from src.models.CNN_basic import SequenceCNN
from sklearn.model_selection import train_test_split
import torch
import pandas as pd


def check_dataloader_output(dataloader):
    for batch in dataloader:
        inputs, targets = batch
        print(f"Inputs: {inputs}")
        print(f"Targets: {targets}")
        # print(f"Inputs Shape: {torch.tensor(inputs).shape}")
        # print(f"Inputs Shape: {inputs.shape}")
        # print(f"Targets Shape: {targets.shape}")
        # print(f"Sample Targets: {targets[:5]}")  # Print first 5 targets for inspection.
        # new = inputs.view(inputs.size(0), -1, 4)
        # print(f"New shape: {new.shape}")
         # Break after the first batch.

data_module = FixedLengthSequenceModule(num_workers=0,
                                        return_type="hmm_match_sequence",
                                        batch_size=2,
                                        dataset=Path("data/processed/10_datasets/phage_25_fixed_25000_reduced_90_ws"),
                                        use_saved=True,
                                        pad_pack=True)
data_module.setup()

# Check train DataLoader
train_dataloader = data_module.train_dataloader()

print("Train DataLoader:")
check_dataloader_output(train_dataloader)

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