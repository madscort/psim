from pathlib import Path
from Bio import SeqIO
import numpy as np
from tempfile import TemporaryDirectory
from src.data.homology_reduction import run_cd_hit

np.random.seed(1)

coordtable = Path("data/processed/01_combined_databases/satellite_coordinates.tsv") # contains sample_id, ref_seq, coord_start, coord_end
ref_seqs = Path("data/processed/01_combined_databases/reference_sequences/")
non_reduced_output = Path("data/processed/06_host_sequences/sampled_host_seqs.fna")
reduce = False
reduced_output = Path("data/processed/06_host_sequences/sampled_host_seqs.90.fna")
identity = 0.9

flank_length = 0
ref_dict = {}

with open(coordtable, "r") as f:
    for line in f:
        line = line.strip().split("\t")
        if line[1] not in ref_dict:
            ref_dict[line[1]] = {}
            start = int(line[2])-flank_length
            if start < 0:
                start = 0
            end = int(line[3])+flank_length
            ref_dict[line[1]]['start'] = [start]
            ref_dict[line[1]]['end'] = [end]
        else:
            start = int(line[2])-flank_length
            if start < 0:
                start = 0
            end = int(line[3])+flank_length
            ref_dict[line[1]]['start'].append(start)
            ref_dict[line[1]]['end'].append(end)

# Get min and max of start and end coordinates.
# Forces algorithm to sample outside of satellite regions.
for ref in ref_dict:
    ref_dict[ref]['start'] = min(ref_dict[ref]['start'])
    ref_dict[ref]['end'] = max(ref_dict[ref]['end'])

with TemporaryDirectory() as tmp:
    
    # Create fasta file with all sequences:
    if reduce:
        raw_host_sample = Path(tmp, "raw_host_sample.fna")
    else: 
        raw_host_sample = non_reduced_output
    
    # Sample X random Y bp sequences from each reference sequence outside of the satellite region
    n_samples = 1
    sample_length = 25000

    with open(raw_host_sample, "w") as f:
        for ref in ref_dict:
            input_file = Path(ref_seqs, f"{ref}.fna")
            for record in SeqIO.parse(input_file, 'fasta'):
                seq_len = len(record.seq)
                right_flank = seq_len - ref_dict[ref]['end']
                left_flank = ref_dict[ref]['start']
                if right_flank < sample_length and left_flank < sample_length:
                    print(f"No space for sampling in {ref}")
                    continue
                # Do either left (0) or right (1) flank:
                flank = np.random.randint(0, 2)
                if right_flank < sample_length:
                    flank = 0
                if left_flank < sample_length:
                    flank = 1
                
                if flank == 0:
                    # Get random start position on left flank
                    start = np.random.randint(0, ref_dict[ref]['start']-sample_length)
                    end = start + sample_length
                    print(f">{ref}_{start}_{end}", file=f)
                    print(record.seq[start:end], file=f)
                if flank == 1:
                    # Get random start position on right flank
                    start = np.random.randint(ref_dict[ref]['end'], seq_len-sample_length)
                    end = start + sample_length
                    # Write to file:
                    print(f">{ref}_{start}_{end}", file=f)
                    print(record.seq[start:end], file=f)
    if reduce:
        # Run CD-HIT-est
        run_cd_hit(raw_host_sample, identity, reduced_output)

        # Parse output
        identifiers = []
        with open(reduced_output, "r") as f:
            for record in SeqIO.parse(f, "fasta"):
                identifiers.append(record.id)
        
        if len(identifiers) == len(set(identifiers)):
            print(f"Reduced sequences to {len(set(identifiers))} with identity {identity}")
        else:
            print(f"Reduced sequences to {len(set(identifiers))} with identity {identity} BUT {len(identifiers) - len(set(identifiers))} duplicates were found!")
                    
            

