import esm
import numpy as np
import pandas as pd
import torch
import sys
from sklearn.decomposition import PCA
from pathlib import Path
import matplotlib.pyplot as plt
from src.data.get_sequence import get_protein_seq_str
from sklearn.model_selection import train_test_split

datafolder = Path("data/processed/10_datasets/phage_25_fixed_25000_reduced_90_ws")

df_sampletable = pd.read_csv(datafolder / "sampletable.tsv", sep="\t", header=None, names=['id', 'type', 'label'])
fit, test = train_test_split(df_sampletable, stratify=df_sampletable['type'], test_size=0.05)
positive_seqs = [get_protein_seq_str(Path(datafolder, "sequences", f"{id}.fna")) for id in test[test['label'] == 1]['id'].values]
negative_seqs = [get_protein_seq_str(Path(datafolder, "sequences", f"{id}.fna")) for id in test[test['label'] == 0]['id'].values]

# Concatenate positive and negative genome sequences and assign labels
genome_data = positive_seqs + negative_seqs
labels = [1] * len(positive_seqs) + [0] * len(negative_seqs)

model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results

genome_vectors = []

for genome in genome_data:
    protein_vectors = []
    
    for protein_seq in genome:
        data = [(f"protein_{idx}", protein_seq) for idx, protein_seq in enumerate(genome)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[6], return_contacts=True)

        token_representations = results["representations"][6]
        protein_vector = token_representations.mean(dim=1) # Mean across tokens
        protein_vectors.append(protein_vector)

    genome_vector = torch.stack(protein_vectors).mean(dim=0) # Mean across proteins
    genome_vectors.append(genome_vector)

genome_vectors = torch.stack(genome_vectors).numpy()

torch.save(genome_vectors, Path("data/visualization/esm_embedding/esm_embeddings.pt"))

# Dimensionality reduction to 2D
pca = PCA(n_components=2)
genome_2d = pca.fit_transform(genome_vectors)

# Visualization
colors = ['red', 'blue'] # For example, 'red' for positive and 'blue' for negative
labels_unique = list(set(labels))

for label, color in zip(labels_unique, colors):
    indices = [i for i, l in enumerate(labels) if l == label]
    plt.scatter(genome_2d[indices, 0], genome_2d[indices, 1], color=color, label=f"Class {label}")

plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend()
plt.title('2D Visualization of Genomes')
plt.show()
plt.savefig("data/visualization/esm_embedding/esm_embed_pca.png")