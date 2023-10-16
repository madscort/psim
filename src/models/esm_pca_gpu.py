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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

datafolder = Path("data/processed/10_datasets/phage_25_fixed_25000_reduced_90_ws")

df_sampletable = pd.read_csv(datafolder / "sampletable.tsv", sep="\t", header=None, names=['id', 'type', 'label'])
fit, test = train_test_split(df_sampletable, stratify=df_sampletable['type'], test_size=0.05)
positive_seqs = [get_protein_seq_str(Path(datafolder, "sequences", f"{id}.fna")) for id in test[test['label'] == 1]['id'].values]
negative_seqs = [get_protein_seq_str(Path(datafolder, "sequences", f"{id}.fna")) for id in test[test['label'] == 0]['id'].values]

genome_data = positive_seqs + negative_seqs
labels = [1] * len(positive_seqs) + [0] * len(negative_seqs)

model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()

model = model.half().to(device)

batch_converter = alphabet.get_batch_converter()
model.eval()

genome_vectors = []

MAX_PROTEINS_PER_GENOME = 1

for n, genome in enumerate(genome_data):
    print("processing: ", n)
    protein_vectors = []
    num_proteins = len(genome)
    
    if num_proteins > MAX_PROTEINS_PER_GENOME:
        genome = genome[:MAX_PROTEINS_PER_GENOME]
        num_proteins = MAX_PROTEINS_PER_GENOME

    for protein_seq in genome:
        data = [("protein", protein_seq)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)
        
        with torch.cuda.amp.autocast():
            results = model(batch_tokens, repr_layers=[12], return_contacts=True)

        token_representations = results["representations"][12]
        protein_vector = token_representations.mean(dim=1).cpu()
        protein_vectors.append(protein_vector)

        # Free up memory
        del results, batch_tokens, token_representations
        torch.cuda.empty_cache()

    genome_vector = torch.stack(protein_vectors).mean(dim=0).squeeze()
    genome_vectors.append(genome_vector)

genome_vectors = torch.stack(genome_vectors).numpy()
torch.save(genome_vectors, Path("data/visualization/esm_embedding/esm_embeddings.pt"))

print(genome_vectors.shape)

pca = PCA(n_components=2)
genome_2d = pca.fit_transform(genome_vectors)

colors = ['red', 'blue']
labels_unique = list(set(labels))

for label, color in zip(labels_unique, colors):
    indices = [i for i, l in enumerate(labels) if l == label]
    plt.scatter(genome_2d[indices, 0], genome_2d[indices, 1], color=color, label=f"Class {label}")

plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend()
plt.title('2D Visualization of Genomes')
plt.show()
plt.savefig("data/visualization/esm_embedding/esm_embeddings_pca_t30.png")
