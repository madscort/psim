from pathlib import Path
from Bio import SeqIO

# Split fasta into individual files using biopython

input_file = Path("data/processed/01_combined_renamed/all_reference_sequences.fna")

# Iterate over each record in the input file and write it to a separate file
for record in SeqIO.parse(input_file, 'fasta'):
    # Replace 'output_folder' with the name of the folder where you want to save the output files
    output_file = f"./data/processed/01_combined_renamed/individual_reference_sequences/{record.id}.fna"
    SeqIO.write(record, output_file, 'fasta')
