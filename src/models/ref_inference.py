from pathlib import Path
from pytorch_lightning import seed_everything
from tqdm import tqdm
from multiprocessing import Pool
from typing import List, Tuple
from tempfile import TemporaryDirectory
import torch
import logging
import sys
import pandas as pd
from torch.nn.functional import one_hot
from Bio import SeqIO
import pyrodigal_gv
import collections
import numpy as np
import torch.nn.functional as F
from src.models.LNSequenceModule import SequenceModule
from src.data.LN_data_module import encode_sequence
from src.data.build_stringDB_pfama import get_one_string
from torch.nn.functional import softmax
from src.data.build_stringDB_novo import HMMER
from src.data.build_stringDB_blitsDB import SMMseqs2


translate = str.maketrans("ACGTURYKMSWBDHVN", "0123444444444444")


def main():
    
    seed_everything(1)

    device = torch.device("cpu")

    # Get all info:
    sampletable = Path("data/processed/01_combined_databases/sample_table.tsv") # contains sample_id, type and label
    sample_table_test = Path("data/processed/10_datasets/dataset_v02/test.tsv") # used to get sequences
    coordtable = Path("data/processed/01_combined_databases/satellite_coordinates.tsv") # contains sample_id, ref_seq, coord_start, coord_end
    ps_sample = collections.namedtuple("ps_sample", ["sample_id", "type", "ref_seq", "coord_start", "coord_end"])

    ref_seqs = Path("data/processed/01_combined_databases/reference_sequences/")
    ps_taxonomy = Path("data/processed/01_combined_databases/ps_tax_info.tsv")
    output_root = Path("data/visualization/sliding_window/version02_protein_version_alldb")
    output_root.mkdir(parents=True, exist_ok=True)
    output_file = output_root / "predictions.tsv"
    test = pd.read_csv(sample_table_test, sep="\t", header=0, names=['id', 'type', 'label'])
    sampleids = test[test['label'] == 1]['id'].tolist()[:50]
    
    model_checkpoint = Path("models/transformer/alldb_v02_small_iak7l6eg.ckpt")
    model_database = Path("data/processed/10_datasets/v02/attachings_allDB/mmseqsprofileDB/profileDB")
    model_database_out = output_root / "db_out"
    model_database_out.mkdir(parents=True, exist_ok=True)

    stride = 25000
    chunk_size = 25000
    transformer = True
    num_threads = 4

    types = {}
    samples = {}
    tax = {}
    with open(ps_taxonomy, "r") as f:
        f.readline()
        for line in f:
            line = line.strip().split("\t")
            try:
                tax[line[0].strip()] = line[3].strip()
            except IndexError:
                tax[line[0].strip()] = "unknown"
                print(line)

    with open(sampletable, "r") as f:
        for sample in f:
            sample = sample.strip().split("\t")
            types[sample[0]] = sample[1]

    with open(coordtable, "r") as f:
        for line in f:
            line = line.strip().split("\t")
            samples[line[0]] = ps_sample(line[0], types[line[0]], line[1], int(line[2]), int(line[3]))
    
    ref_dict = {}
    with open(coordtable, "r") as f:
        for line in f:
            line = line.strip().split("\t")
            if line[1] not in ref_dict:
                ref_dict[line[1]] = []
                start = int(line[2])
                end = int(line[3])
                ref_dict[line[1]].append((start,end))
            else:
                start = int(line[2])
                end = int(line[3])
                ref_dict[line[1]].append((start,end))

    analysed_seqs = set()
    sequences = []
    
    print(len(sampleids))
    GET_TRUTH = False
    if GET_TRUTH:
        overlaps = 0
        for n, id in enumerate(sampleids):
            if n % 10 == 0:
                print(n)
            ref_seq = samples[id].ref_seq
            if ref_seq in analysed_seqs:
                continue
            analysed_seqs.add(ref_seq)
            coords = sorted(ref_dict[ref_seq])
            # Check for overlapping coords in list:
            overlap = False

            for n, coord_set in enumerate(coords):
                if n == 0:
                    continue
                if coord_set[0] < coords[n-1][1]:
                    print(ref_seq)
                    print(coords)
                    print("Overlapping coordinates")
                    overlap = True
                    overlaps += 1
            if overlap:
                continue

            # Get sequence:
            seq = ""
            with open(Path(ref_seqs, ref_seq + ".fna"), "r") as f:
                f.readline()
                for line in f:
                    seq += line.strip()
            
            start = 0
            length = len(seq)
            true_seq = []
            for coord_set in coords:
                coord_start = coord_set[0]
                coord_end = coord_set[1]
                before = [0]*(coord_start-start)
                sat = [1]*(coord_end-coord_start)
                true_seq += before + sat
                start = coord_end
            true_seq += [0]*(length-start)

            true_sum = 0
            for coord_set in coords:
                true_sum += coord_set[1]-coord_set[0]
            if true_sum != sum(true_seq):
                print(ref_seq)
                print(true_sum, sum(true_seq))
                print(coords)
                sys.exit("Satellite lengths do not match")
            if len(seq) != len(true_seq):
                print(ref_seq)
                print(coords)
                print(len(seq), len(true_seq))
                sys.exit("Ref lengths do not match")
            
            # Slide over sequence:
            true_chunks = [true_seq[i:i+chunk_size] for i in range(0, len(true_seq)-chunk_size+1, stride)]
            
            # Get predictions:
            truths = []
            for true_chunk in true_chunks:
                if 1 in true_chunk:
                    truths.append(1)
                else:
                    truths.append(0)
            
            sequences.append(Sequence(ref_seq, tax[id], seq, truths))
        
        print("Overlaps: ", overlaps)
        torch.save(sequences, output_root / "sequences.pt")
    else:
        sequences = torch.load(output_root / "sequences.pt")
    
    TRANSLATE = False
    if TRANSLATE:
        logging.info("Start translation")
        translated_sequences = parallel_gene_prediction(sequences, output_root / "proteins.faa", 4)
        logging.info("Finished translation")
        torch.save(translated_sequences, output_root / "proteins.pt")
    else:
        translated_sequences = torch.load(output_root / "proteins.pt")

    model = SequenceModule.load_from_checkpoint(checkpoint_path=model_checkpoint,
                                                map_location=device)
    model.to(device)
    model.eval()
    
    if transformer:
        vocab_map = model.vocab_map
        max_seq_length = 56 # LATER ADD TO: model.max_seq_length

    database_type = "mmseqs"
    database_source = "pfama" 
    if database_type == "hmm":
        mDB = HMMER(hmm=model_database, db=output_root / "proteins.faa", out=model_database_out, scan=True)
    elif database_type == "mmseqs":
        mDB = SMMseqs2(input_fasta=output_root / "proteins.faa", mmseqs_db=model_database, output_dir=model_database_out)

    CALL_ALL = True
    if CALL_ALL:
        if database_type == "hmm":
            mDB.scan()
        elif database_type == "mmseqs":
            mDB.run_alignment(threads=num_threads, evalue=1e-3)

    hits = mDB.get_best_hits()
    
    print("Vocabulary size: ", len(vocab_map))
    print("Markers used: ", len(set(x.target_name for x in hits.values())))

    ecoli = {"NZ_CP013662.1",
        "NZ_CP019961.1",
        "NZ_CP022664.1",
        "NZ_CP024278.1",
        "NZ_CP028741.1",
        "NZ_CP051735.1",
        "NZ_CP061101.1",
        "NZ_CP068802.1",
        "NZ_CP091704.1"
        }

    all_preds = []
    all_preds_prob = []
    protein_stride = 1
    for n, translated_sequence in enumerate(translated_sequences):
        # if sequences[n].id not in ecoli:
        #     continue
        
        if n % 10 == 0:
            print(n)
        print(translated_sequence[0].origin)
        predictions = [0]*len(sequences[n].seq)
        # Go through sequence in strides of stride size:
        for i in range(0, len(translated_sequence)-protein_stride+1, protein_stride):

            # Eat proteins from index until chunk_size is reached:
            eating_proteins = True
            current_index = i
            length_dna = 0
            model_string = []
            protein_string = []
            while eating_proteins and current_index < len(translated_sequence) and len(model_string) < max_seq_length:
                if length_dna + len(translated_sequence[current_index].seq) < chunk_size:
                    length_dna += len(translated_sequence[current_index].seq)
                    if translated_sequence[current_index].protein in hits:
                        if database_type == "hmm":
                            if database_source == "pfama":
                                model_string.append(hits[translated_sequence[current_index].protein].target_accession)
                            else:
                                model_string.append(hits[translated_sequence[current_index].protein].target_name)
                        elif database_type == "mmseqs":
                            target_name = hits[translated_sequence[current_index].protein].target_name.replace("|", "_")
                            model_string.append(target_name)
                    else:
                        model_string.append('no_hit')
                    protein_string.append((translated_sequence[current_index].start, translated_sequence[current_index].end))
                    current_index += 1
                else:
                    eating_proteins = False
            if transformer:
                # Get sequence of HMM profile matches and pad
                encoded_string = torch.tensor(encode_sequence(model_string, vocab_map))
                padding = max_seq_length - len(encoded_string)
                encoded_string = F.pad(encoded_string, (0, padding), "constant", 0)
                sequence = encoded_string.unsqueeze(0)
            prediction = model(sequence)
            if prediction.argmax(dim=1).tolist()[0] == 1:
                for x in range(protein_string[0][0], protein_string[-1][1]):
                    predictions[x] = 1
            # probabilities.append(softmax(prediction, dim=1)[0][1].item())
        # Slide over sequence:
        predict_chunks = [predictions[i:i+chunk_size] for i in range(0, len(predictions)-chunk_size+1, stride)]
        
        # Get predictions:
        predict_bins = []
        for true_chunk in predict_chunks:
            if 1 in true_chunk:
                predict_bins.append(1)
            else:
                predict_bins.append(0)
        all_preds.append(predict_bins)

    # with open(output_file, "w") as f:
    #     # Go through each reference sequence
    #     cin = -1
    #     for i in range(len(sequences)):
    #         if sequences[i].id not in ecoli:
    #             continue
    #         cin += 1
    #         # Go through the actual sequence
    #         for loc in range(len(all_preds[cin])):
    #             print(loc+1,                    # location
    #                   sequences[i].id,             # reference sequence id
    #                   sequences[i].host,             # reference sequence species
    #                   sequences[i].true_seq[loc],       # true value at location
    #                   all_preds[cin][loc],            # predicted value at location
    #                   "0",   # probability of prediction at location, add later: all_preds_prob[i][loc]
    #                   sep="\t",
    #                   file=f)

    with open(output_file, "w") as f:
        # Go through each reference sequence
        for i in range(len(sequences)):
            # Go through the actual sequence
            for loc in range(len(all_preds[i])):
                print(loc+1,                    # location
                      sequences[i].id,             # reference sequence id
                      sequences[i].host,             # reference sequence species
                      sequences[i].true_seq[loc],       # true value at location
                      all_preds[i][loc],            # predicted value at location
                      "0",   # probability of prediction at location, add later: all_preds_prob[i][loc]
                      sep="\t",
                      file=f)

Protein = collections.namedtuple("Protein", ["origin", "protein", "position", "start", "end", "seq"])
Sequence = collections.namedtuple("Sequence", ["id", "host", "seq", "true_seq"])
Hit = collections.namedtuple('Hit', ['target_name', 'target_accession', 'bitscore'])

def gene_prediction(seq_obj: Sequence) -> List[Protein]:
    proteins = []
    orf_finder = pyrodigal_gv.ViralGeneFinder(meta=True, mask=True)
    for n, pred in enumerate(orf_finder.find_genes(seq_obj.seq)):
        proteins.append(Protein(origin=seq_obj.id,
                                protein=f"{seq_obj.id}|{n}",
                                position=n,
                                start=pred.begin,
                                end=pred.end,
                                seq=pred.translate(include_stop=False)))
    return proteins

def write_proteins_to_file(sequence: List[Protein], f_out):
    for protein in sequence:
        print(f">{protein.protein}", file=f_out)
        print(protein.seq, file=f_out)

def parallel_gene_prediction(sequences: List[Tuple], output: Path, threads: int) -> List:
    with Pool(threads) as pool:
        results = pool.map(gene_prediction, sequences)
    with open(output, "w") as f_out:
        for sequence in results:
            write_proteins_to_file(sequence, f_out)
    return results


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
