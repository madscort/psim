from pathlib import Path
import sys
### Clean taxonomic data and remove all whitespace

tax_file = Path("data/processed/01_combined_databases/ps_tax_info.tsv")
clean_file = Path("data/processed/03_taxonomical_annotation/ps_tax_info.tsv")

with open(tax_file, "r") as fin, open(clean_file, "w") as fout:
    for line in fin:
        line = [x.strip() for x in line.split("\t")]
        line = [x.replace(" ", "_") for x in line]
        line = [line[0]] + [x.lower() for x in line[1:]]
        print("\t".join(line), file=fout)