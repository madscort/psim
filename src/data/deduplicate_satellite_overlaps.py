from pathlib import Path
import collections

# mads 2023-10-23
# Script for removing overlapping satellite sequences
# Keeps the sequence from Urvish database if duplicate between Urvish vs Rocha
# Keeps the smaller sequence in any other case
#
# Takes a sampletable and a coordtable as input
# Outputs a new sampletable with deduplicated samples

def resolve_overlap(coord1, coord2):
    # Check if either coordinate set's sample has "U"
    u_in_coord1 = "U" in coord1[2]
    u_in_coord2 = "U" in coord2[2]

    # If one has "U" but not the other, keep the one with "U"
    if u_in_coord1 and not u_in_coord2:
        return coord1
    elif u_in_coord2 and not u_in_coord1:
        return coord2

    # If neither or both have "U", keep the smaller coordinate set
    size_coord1 = coord1[1] - coord1[0]
    size_coord2 = coord2[1] - coord2[0]

    return coord1 if size_coord1 <= size_coord2 else coord2

input_sampletable = Path("data/processed/01_combined_databases/sample_table.tsv")
coordtable = Path("data/processed/01_combined_databases/satellite_coordinates.tsv")
output_sampletable = Path("data/processed/02_preprocessed_database/01_deduplication/sampletable.tsv")
ps_sample = collections.namedtuple("ps_sample", ["sample_id", "type", "label"])

with open(input_sampletable, "r") as f:
    sample_check = set()
    samples = {}
    for line in f:
        line = line.strip().split("\t")
        samples[line[0]] = ps_sample(line[0], line[1], line[2])
        sample_check.add(line[0].strip())

ref_dict = {}
with open(coordtable, "r") as f:
    for line in f:
        line = line.strip().split("\t")
        start = int(line[2])
        end = int(line[3])
        sample_id = line[0]
        sample_check.remove(sample_id)
        if line[1] not in ref_dict:
            ref_dict[line[1]] = []
            ref_dict[line[1]].append((start,end,sample_id))
        else:
            ref_dict[line[1]].append((start,end,sample_id))

if len(sample_check) > 0:
    print("Samples not found in coordtable:")
    print(samples)

overlaps = 0
total_ps = 0
keep_ref_dict = {}

for ref in ref_dict:
    coords = sorted(ref_dict[ref])
    total_ps += len(coords)
    if len(ref_dict[ref]) == 1:
        keep_ref_dict[ref] = ref_dict[ref]
        continue
    keep_coords = []
    for n, coord_set in enumerate(coords):
        if n == 0:
            keep_coords.append(coord_set)
            continue

        # If overlapping
        if coord_set[0] < keep_coords[-1][1]:
            overlaps += 1
            keep_coord = resolve_overlap(coord_set, keep_coords[-1])
            # Replace the last added coordinate with the resolved coordinate
            keep_coords[-1] = keep_coord
        else:
            keep_coords.append(coord_set)

    keep_ref_dict[ref] = keep_coords

total_ps_keep = 0
for ref in keep_ref_dict:
    total_ps_keep += len(keep_ref_dict[ref])

with open(output_sampletable, "w") as f:
    for ref in keep_ref_dict:
        for coord_set in keep_ref_dict[ref]:
            f.write(f"{coord_set[2]}\t{samples[coord_set[2]].type}\t1\n")

print("Raw PS: ", total_ps)
print("Overlaps: ", overlaps)
print("Raw minus overlaps: ", total_ps-overlaps)
print("PS after deduplication", total_ps_keep)
