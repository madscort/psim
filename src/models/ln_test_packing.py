from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from src.data.LN_data_module import FixedLengthSequenceModule
from src.models.CNN_basic import SequenceCNN
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
from torch.nn.utils.rnn import pad_packed_sequence
from src.models.LSTM_collection import BasicLSTM

def check_dataloader_output(dataloader):
    for batch in dataloader:
        sequences_packed, lengths, labels = batch
        print(f"Inputs (packed): {sequences_packed}")

        # You might also want to unpack the sequences to print them
        inputs, _ = pad_packed_sequence(sequences_packed, batch_first=True)
        print(f"Inputs (unpacked): {inputs}")
        print(f"Sequence Lengths: {lengths}")

        print(f"Targets: {labels}")
        print(f"Targets Shape: {labels.shape}")

        break  # Break after the first batch


data_module = FixedLengthSequenceModule(num_workers=0, return_type="gc_sequence", batch_size=64, pad_pack=True)
data_module.setup()

# Check train DataLoader
train_dataloader = data_module.train_dataloader()

# print("Train DataLoader:")
# check_dataloader_output(train_dataloader)

# # Check val DataLoader (if defined in your DataModule)
# if hasattr(data_module, 'val_dataloader') and callable(data_module.val_dataloader):
#     val_dataloader = data_module.val_dataloader()
#     print("\nValidation DataLoader:")
#     check_dataloader_output(val_dataloader)

# # Check test DataLoader (if defined in your DataModule)
# if hasattr(data_module, 'test_dataloader') and callable(data_module.test_dataloader):
#     test_dataloader = data_module.test_dataloader()
#     print("\nTest DataLoader:")
#     check_dataloader_output(test_dataloader)

# Initialize the model
model = BasicLSTM(input_size=1)

# Initialize an instance of BasicLSTM and perform a forward pass test on the first batch of the train dataloader
for batch in train_dataloader:
    sequences_packed, labels = batch
    outputs = model(sequences_packed)
    print(f"Model outputs: {outputs}")
    break