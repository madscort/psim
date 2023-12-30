from Bio import SeqIO
from pathlib import Path

output_folder = Path("data/visualization/profile_function/transformer")
output_folder.mkdir(parents=True, exist_ok=True)
output_file = output_folder / "top_21_features_freq_adj.faa"
in_fasta = Path("data/processed/10_datasets/v02/attachings_allDB/proteins.faa").absolute()
### These are the frequence adjusted features from the transformer model
features = ["PS_U825_6", "PS_R4849_22","PS_R4861_12","PS_R3488_7","PS_R1893_9","PS_R162_13","PS_U2336_31","PS_U125_12",
    "PS_U522_8","PS_R3928_8", "PS_U1789_17","NZ_CP046669.1_2830419_2855419_19","PS_R3928_11","PS_R289_14",
    "PS_U1210_7", "PS_R3439_21","PS_R482_1","PS_U828_22", "PS_R4188_21","PS_R91_24","PS_U804_13"]

### These are the top raw positive only features from the transformer model
features_2 = ["PS_R3011_14","PS_R3009_27","PS_R3408_17", 
"PS_U1263_20","PS_U2336_31","PS_R2991_31", 
"PS_R2231_6", "PS_U2199_21","PS_R3921_15", 
"IMGVR_UViG_2648501149_000023_1",  "PS_R2988_1", "PS_R23_27", 
"PS_R3488_7", "NZ_CP046669.1_2830419_2855419_19","PS_R4861_12", 
"PS_R4236_11","PS_R3928_8", "PS_U825_6", 
"PS_R3928_11","PS_R162_13",  
]


other_reps = Path("data/visualization/profile_function/linear_c_004_annotations.tsv")

names = set()
with open(other_reps, 'r') as fin:
    for line in fin:
        names.add(line.strip().replace('|','_').split("\t")[0])

print("Difference in names: ", len(names.difference(set(features))))
print("Difference in names: ", len(names.difference(set(features_2))))

# with open(output_file, "w") as fasta:
#     for feature in features:
#         for seq_record in SeqIO.parse(in_fasta, "fasta"):
#             if seq_record.id.replace('|','_') == feature:
#                 SeqIO.write(seq_record, fasta, "fasta")
#                 break
#         else:
#             print(f"ERROR: {feature} not found in fasta file")
