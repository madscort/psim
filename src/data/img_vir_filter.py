from pathlib import Path
import sys
import collections
import numpy as np

np.random.seed(1)

img_vr_host_info = Path("data/external/databases/IMG_VR/IMGVR_all_Host_information-high_confidence.tsv")
img_vr_seq_info = Path("data/external/databases/IMG_VR/IMGVR_all_Sequence_information-high_confidence.tsv")
filter_id = Path("data/processed/10_datasets/phage_25_fixed_25000_reduced_90_ws/sampletable.tsv")
filter_taxonomy = Path("data/processed/01_combined_renamed/ps_tax_info.tsv")
filter_imgvr_seqs_list = Path("data/processed/10_datasets/phage_25_fixed_25000_reduced_90_ws/IMGVR_all_filtered_seqs_info_wcoord.tsv")
# # Filter by species

# Get list of species from sequences in sampletable.
# Get all valid satellite ids:
satellite_ids = []

with open(filter_id, "r") as f:
    for line in f:
        satellite_ids.append(line.split("\t")[0].strip())

if len(satellite_ids) != len(set(satellite_ids)):
    print("Duplicate satellite ids in sampletable")
    sys.exit()

# Get species by satellite id:
satellite_species = {}
species_imgvr_count = {}

with open(filter_taxonomy, "r") as f:
    for line in f:
        if line.split("\t")[0].strip() not in satellite_ids:
            continue
        try:
            species = line.split("\t")[3].strip()
        except IndexError:
            print("Warning, missing taxinfo: ", line)
            continue
        if species in satellite_species:
            satellite_species[species] += 1
        else:
            satellite_species[species] = 1
            species_imgvr_count[species] = 0

## Extract sequences by species

img_seq = set()
img_seq_species = {}

with open(img_vr_host_info, "r") as f:
    f.readline()
    for line in f:
        sequence = line.split("\t")[0].strip()
        type = line.split("\t")[1]
        host = line.split("\t")[2].split(";")
        family, genus, species = host[4][3:].strip(), host[5][3:].strip(), host[6][3:].strip()
        if family == "" and genus == "" and species == "":
            continue
        #print(f"{sequence}\t{type}\t{family}\t{genus}\t{species}")
        if species in satellite_species:
            species_imgvr_count[species] += 1
            img_seq.add(sequence)
            img_seq_species[sequence] = species

missed_sp = 0
missed_seq = 0

for species in species_imgvr_count:
    if species_imgvr_count[species] == 0:
        missed_sp += 1
        missed_seq += satellite_species[species]
    print(species.ljust(40),f"{species_imgvr_count[species]}".ljust(10),satellite_species[species])

print(f"Missed species: {missed_sp}")
print(f"Missed sequences: {missed_seq}")

high_quality = 0
reference = 0
total = 0
types = {}

# Save sequences with species, type and coordinates
with open(filter_imgvr_seqs_list, "w") as o:
    with open(img_vr_seq_info, "r") as f:
        for line in f:
            sequence = line.split("\t")[0].strip()
            if sequence in img_seq:
                total += 1
                type = line.split("\t")[7].strip()
                coord = line.split("\t")[3].strip()
                if coord == "whole":
                    coord_start = 0
                    coord_end = -1
                else:
                    coord_start = int(coord.split("-")[0])
                    coord_end = int(coord.split("-")[1])
                if type not in types:
                    types[type] = 0
                else:
                    types[type] += 1
                if line.strip().split("\t")[12] == "Reference":
                    reference += 1
                if line.strip().split("\t")[12] == "High-quality":
                    high_quality += 1
                print(f"{sequence}\t{img_seq_species[sequence]}\t{type}\t{coord_start}\t{coord_end}", file=o)

print(f"Total: {total}")
print(f"Reference: {reference}")
print(f"High quality: {high_quality}")
print(f"Total high quality: {reference + high_quality}")
for type in types:
    print(type, types[type])

Prosequence = collections.namedtuple("Prosequence", ["species", "type", "coord_start", "coord_end"])

fixed_deduplicated = Path("data/processed/05_viral_sequences/imgvr_filtered/IMGVR_all_filtered_seqs_rmdup_min_25000_max_50000.fna")
reduced_imgvr_seqs_list = Path("data/processed/10_datasets/phage_25_fixed_25000_reduced_90_ws/IMGVR_minimal_seq_list.tsv")

sequence_info = {}
species_sequences = {}
for species in satellite_species:
    species_sequences[species] = []

# Load sequence info
with open(filter_imgvr_seqs_list, "r") as f:
    for line in f:
        item = line.strip()
        seqid = item.split("\t")[0]
        species = item.split("\t")[1]
        type = item.split("\t")[2]
        coord_start = int(item.split("\t")[3])
        coord_end = int(item.split("\t")[4])
        sequence_info[seqid] = Prosequence(species, type, coord_start, coord_end)

# Load sequences in deduplicated set
sequences = set()
with open(fixed_deduplicated, "r") as f:
    for line in f:
        if line.startswith(">"):
            sequences.add(line.strip()[1:])

provirus = 0

# Create dictionary of species to sequences
for sequence in sequences:
    if sequence not in sequence_info:
        print(f"Missing sequence info for {sequence}")
        sys.exit()
    species_sequences[sequence_info[sequence].species].append(sequence)

uncovered = 0

for species in species_sequences:
    if len(species_sequences[species]) == 0:
        uncovered += 1
    print(species.ljust(40), len(species_sequences[species]))

print(f"Uncovered species: {uncovered}")
print(f"Uncovered percentage: {uncovered/len(species_sequences)}")

low = set(["Enterobacter roggenkampii",
           "Escherichia albertii",
           "Escherichia fergusonii",
           "Klebsiella grimontii",
           "Klebsiella michiganensis",
           "Klebsiella quasipneumoniae",
           "Lacticaseibacillus paracasei",
           "Lacticaseibacillus rhamnosus",
           "Raoultella ornithinolytica",
           "Shigella flexneri",
           "Staphylococcus epidermidis"])
med = set(["Enterobacter hormaechei",
           "Lactiplantibacillus plantarum",
           "Serratia marcescens"])
high = set(["Citrobacter freundii",
            "Escherichia coli",
            "Klebsiella pneumoniae",
            "Salmonella enterica",
            "Staphylococcus aureus"])
default_seq_count = 5
low_seq_count = 100
med_seq_count = 250
high_seq_count = 500

# Get count of random sequences for each species

with open(reduced_imgvr_seqs_list, "w") as o:
    for species in species_sequences:
        output_seqs = []
        if len(species_sequences[species]) == 0:
            continue
        if species in low:
            if len(species_sequences[species]) < low_seq_count:
                output_seqs = species_sequences[species]
            else:
                output_seqs = np.random.choice(species_sequences[species], low_seq_count, replace=False)
        elif species in med:
            if len(species_sequences[species]) < med_seq_count:
                output_seqs = species_sequences[species]
            else:
                output_seqs = np.random.choice(species_sequences[species], med_seq_count, replace=False)
        elif species in high:
            if len(species_sequences[species]) < high_seq_count:
                output_seqs = species_sequences[species]
            else:
                output_seqs = np.random.choice(species_sequences[species], high_seq_count, replace=False)
        else:
            if len(species_sequences[species]) < default_seq_count:
                output_seqs = species_sequences[species]
            else:
                output_seqs = np.random.choice(species_sequences[species], default_seq_count, replace=False)
        for seq in output_seqs:
            print(seq, sep="\t", file=o)

