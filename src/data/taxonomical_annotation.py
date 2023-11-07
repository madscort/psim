from pathlib import Path
import subprocess
import json
import numpy as np
import pytaxonkit as pytk
import re
from tqdm import tqdm
from joblib import Parallel, delayed
import sys
# mads 2023-09-28
# Get tax info for all satellite sequences

def taxid_from_ncbi(genome_id: str):
    cmd = ['datasets', 'summary', 'genome', 'accession', genome_id, '--as-json-lines']
    result = subprocess.run(cmd, stdout=subprocess.PIPE)
    try:
        return json.loads(result.stdout)['organism']['tax_id']
    except json.JSONDecodeError:
        print(f"Failed to decode JSON for genome ID: {genome_id}")
        return None

def taxid_from_ncbi_parallel(genome_acc: list, num_processes: int):
    return Parallel(n_jobs=num_processes)(delayed(taxid_from_ncbi)(id) for id in tqdm(genome_acc))


rocha_meta = Path("data/raw/01_phage_satellites/metadata/satellites_coordinates_ongenome.tsv")
sampletable = Path("data/processed/01_combined_renamed/sample_table.tsv")
ncbi_taxdump = Path("data/external/databases/ncbi_taxdump/")
header_map_file = Path("data/processed/01_combined_renamed/renaming_map.tsv")
psID_header_map = {}
header_psID_map = {}

with open(header_map_file, "r") as f:
    for line in f:
        psID_header_map[line.split("\t")[0]] = line.split("\t")[1].strip()

with open(header_map_file, "r") as f:
    for line in f:
        header_psID_map[line.split("\t")[1].strip()] = line.split("\t")[0]


if __name__ == '__main__':
        ps_ids = []
        # Get taxids and labels for urvish data
        with open(sampletable, "r") as f:
            organisms = []
            for line in f:
                if line.startswith("PS_U"):
                    ps_id = line.split("\t")[0].strip()
                    # Get species name
                    header = psID_header_map[ps_id]
                    header = header.strip()
                    header = re.sub(r'\*\d+', '', header)
                    if ps_id in ps_ids:
                        print(f"Duplicate PS ID: {ps_id}")
                        print(f"Header: {header}")
                        sys.exit()
                    ps_ids.append(ps_id)
                    organisms.append(" ".join(header.split(" ")[1:3]))
            result = pytk.name2taxid(organisms, data_dir=ncbi_taxdump)
            urvish_taxids = result['TaxID'].values
        with open(rocha_meta, "r") as f:
            f.readline()
            genome_acc = []
            for line in f:
                genome_acc.append(line.split("\t")[1])
                ps_ids.append(header_psID_map[line.split("\t")[0]])
        rocha_taxids = taxid_from_ncbi_parallel(genome_acc, -1)
        taxids = np.concatenate([urvish_taxids, np.array(rocha_taxids)])
        lineage = pytk.lineage(taxids, formatstr='{f};{g};{s}', data_dir=ncbi_taxdump)
        with open("data/processed/01_combined_renamed/ps_tax_info.tsv", "w") as f:
            f.write("ps_id\tfamily\tgenus\tspecies\n")
            for i in range(0,len(ps_ids)):
                print(ps_ids[i], "\t", "\t".join(str(lineage['Lineage'].values[i]).split(";")), file=f)
