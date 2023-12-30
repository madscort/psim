from pathlib import Path
import torch
import pandas as pd

# five counts: unknown, metagenome, provirus, host, satellite
# based on type.
# So output: id, type, unknown, metagenome, provirus, host, satellite

dataset_root = Path("data/processed/10_datasets/")
version = "dataset_v02"
dataset = dataset_root / version
data_splits = torch.load(dataset / "strings" / "allDB" / "dataset.pt", map_location="cpu")
outfn = Path("data/visualization/marker_gene_type_dist/v02/allDB.tsv")
outfn.parent.mkdir(parents=True, exist_ok=True)
splits = ['train', 'val', 'test']
with open(outfn, "w") as fout:
    for split in splits:
        df = pd.read_csv(dataset / f"{split}.tsv", sep="\t", header=0, names=["id", "type", "label"])
        sequences = data_splits[split]['sequences']
        test_ids = df["id"].values.tolist()
        test_labels = df["label"].values.tolist()
        test_types = df["type"].values.tolist()

        for n, sequence in enumerate(sequences):
            # Get origin of sequence
            type_id = test_types[n]
            sample_id = test_ids[n]

            if type_id.startswith("prov"):
                origin = "provirus"
            elif type_id.startswith("host"):
                origin = "host"
            elif type_id.startswith("meta"):
                origin = "metagenome"
            else:
                origin = "satellite"

            sat_c = 0
            pro_c = 0
            host_c = 0
            meta_c = 0
            unk_c = 0
            for protein in sequence:
                if protein.startswith("IMGVR"):
                    pro_c += 1
                elif protein.startswith("PS"):
                    sat_c += 1
                elif protein.startswith("N") or protein.startswith("D") or protein.startswith("C"):
                    host_c += 1
                elif protein.startswith("S"):
                    meta_c += 1
                else:
                    unk_c += 1
            
            seq_types = {'satellite': sat_c,
             'provirus': pro_c,
             'host': host_c,
             'metagenomic': meta_c,
             'unknown': unk_c}
            for st in seq_types:
                print(split,
                      sample_id,
                      origin,
                      test_labels[n],
                      st,
                      seq_types[st],
                      sep="\t",
                      file=fout)
