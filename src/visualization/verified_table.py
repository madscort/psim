from pathlib import Path
from collections import Counter

satellite_finder_results = Path("data/visualization/validated_data/satellite_finder/satellite_finder.tsv")
transformer_results = Path("data/visualization/validated_data/transformer/coordinates.tsv")
outf = Path("data/visualization/validated_data/verified_sapis_prediction.tsv")

finder_counts = {}
with open(satellite_finder_results) as fin:
    for line in fin:
        id, strain, count = line.strip().split("\t")
        count = int(count)
        finder_counts[strain] = count

trans_ids = []
with open(transformer_results) as fin:
    for line in fin:
        id = line.strip().split("\t")[0]
        trans_ids.append(id)
trans_counts = Counter(trans_ids)
found = False
with open(outf, "w") as fout:
    for sat in finder_counts:
        id = sat
        if id.startswith("S"):
            if not found:
                id = "ATCC 15305 v1"
                found = True
            else:
                id = "ATCC 15305 v2"
        if id == 'NCTC':
            id = 'NCTC 8325'
        print(id, trans_counts[sat], finder_counts[sat], sep="\t", file=fout)

