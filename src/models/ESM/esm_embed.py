import esm
import numpy as np
import torch
import sys

model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)

data = [
    ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
    ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE")
]

batch_labels, batch_strs, batch_tokens = batch_converter(data)
batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

# Extract per-residue representations (on CPU)
with torch.no_grad():
    results = model(batch_tokens.to(device), repr_layers=[6], return_contacts=True)

token_representations = results["representations"][6]

# Generate per-sequence representations via averaging

sequence_representations = []
for i, tokens_len in enumerate(batch_lens):
    sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))

for seq in sequence_representations:
    print(len(seq))

# # Look at the unsupervised self-attention map contact predictions
# import matplotlib.pyplot as plt
# for (_, seq), tokens_len, attention_contacts in zip(data, batch_lens, results["contacts"]):
#     plt.matshow(attention_contacts[: tokens_len, : tokens_len])
#     plt.title(seq)
#     plt.savefig("data/visualization/test_esm_embed.png")