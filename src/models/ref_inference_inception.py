from pathlib import Path
from pytorch_lightning import seed_everything
from tqdm import tqdm
import torch
import sys
import pandas as pd
from torch.nn.functional import one_hot
import collections
from src.models.LNSequenceModule import SequenceModule
from src.data.LN_data_module import encode_sequence
from src.data.build_stringDB_pfama import get_one_string
from torch.nn.functional import softmax
translate = str.maketrans("ACGTURYKMSWBDHVN", "0123444444444444")


def main():
    
    seed_everything(1)

    device = torch.device("cpu")

    # Get all info:
    sampletable = Path("data/processed/01_combined_databases/sample_table.tsv") # contains sample_id, type and label
    coordtable = Path("data/processed/01_combined_databases/satellite_coordinates.tsv") # contains sample_id, ref_seq, coord_start, coord_end
    ps_sample = collections.namedtuple("ps_sample", ["sample_id", "type", "ref_seq", "coord_start", "coord_end"])
    ref_seqs = Path("data/processed/01_combined_databases/reference_sequences/")
    ps_taxonomy = Path("data/processed/01_combined_databases/ps_tax_info.tsv")
    output_file = Path("data/visualization/sliding_window/predictions_version02_inception_whatstride.tsv")
    stride = 5000
    chunk_size = 25000
    transformer = False

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


    # Get all host sampled seqs instead:

    # with open(Path("data/processed/06_sampled_host_seqs/host_sampled_seqs.fna"), "r") as f:
    #     # Append all sequences to list:
    #     f.readline()
    #     seq = ""
    #     for line in f:
    #         if line.startswith(">"):
    #             seqs.append(seq)
    #             seq = ""
    #             continue
    #         seq += line.strip()
    #     seqs.append(seq)

    seqs = []
    seq_host = []
    all_truths = []

    sample_table_test = Path("data/processed/10_datasets/dataset_v02/test.tsv")
    test = pd.read_csv(sample_table_test, sep="\t", header=0, names=['id', 'type', 'label'])
    # Get sampleids from val for label == 1:

    sampleids = test[test['label'] == 1]['id'].tolist()[:50]
    test_ref_seqs = []
    print(len(sampleids))
    overlaps = 0
    for n, id in enumerate(sampleids):
        if n % 10 == 0:
            print(n)
        ref_seq = samples[id].ref_seq
        if ref_seq in test_ref_seqs:
            continue
        
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
        test_ref_seqs.append(ref_seq)
        seq_host.append(tax[id])

        # Get sequence:
        seq = ""
        with open(Path(ref_seqs, ref_seq + ".fna"), "r") as f:
            f.readline()
            for line in f:
                seq += line.strip()
        seqs.append(seq)
        
        # Create mock "true" sequence:
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
        all_truths.append(truths)
    print("Overlaps: ", overlaps)
    # Get all validation seqs:

    # with open("data/raw/04_verified_sapis/01_seqs_with_host/all_sapi_w_host.fasta", "r") as f:
    #     f.readline()
    #     seq = ""
    #     for line in f:
    #         if line.startswith(">"):
    #             seqs.append(seq)
    #             seq = ""
    #             continue
    #         seq += line.strip().upper()
    #     seqs.append(seq)


    # # Get meta sequences instead:
    # seqs = []
    # with open(Path("data/processed/04_metagenomic_contigs/background/combined_min_200000.fa"), "r") as f:
    #     # Append all sequences to list:
    #     f.readline()
    #     seq = ""
    #     for line in f:
    #         if line.startswith(">"):
    #             seqs.append(seq)
    #             seq = ""
    #             continue
    #         seq += line.strip()
    #     seqs.append(seq)

    model = SequenceModule.load_from_checkpoint(checkpoint_path=Path("models/inception/checkpoint/24eamlea_version02.ckpt").absolute(),
                                                map_location=device)
    model.to(device)

    model.eval()
    
    if transformer:
        vocab_map = model.vocab_map
    seq_preds = []
    all_preds = []
    all_preds_prob = []
    for n, seq in enumerate(seqs):
        print("Sequence: ", n)
        if n % 10 == 0:
            print(n)
        # Create sliding window of size 25000 with adjustable stride:

        # with equal chunk sizes: 
        chunks = [seq[i:i+chunk_size] for i in range(0, len(seq)-chunk_size+1, stride)]
        # with short chunks in end:
        # chunks = [seq[i:i+chunk_size] for i in range(0, len(seq), stride)]
        # Get predictions:
        predictions = []
        probabilities = []
        
        for chunk in tqdm(chunks):
            if transformer:
                model_string = get_one_string(chunk, Path("data/processed/10_datasets/attachings/hmms/hmms.hmm"))
                encoded_string = torch.tensor(encode_sequence(model_string, vocab_map))
                sequence = {"seqs": encoded_string.unsqueeze(0)}
            else:
                sequence = torch.tensor([int(base.translate(translate)) for base in chunk], dtype=torch.float)
                sequence = one_hot(sequence.to(torch.int64), num_classes=5).to(torch.float).permute(1, 0).unsqueeze(0)
            prediction = model(sequence)
            predictions.append(prediction.argmax(dim=1).tolist()[0])
            probabilities.append(softmax(prediction, dim=1)[0][1].item())

        if sum(predictions) > 5:
            seq_preds.append(1)
            print(1)
        else:
            print(0)
            seq_preds.append(0)
        # if 1 in predictions:
        #     seq_preds.append(1)
        #     print(1)
        # else:
        #     print(0)
        #     seq_preds.append(0)
        all_preds.append(predictions)
        all_preds_prob.append(probabilities)

    print((len(seq_preds)-sum(seq_preds))/len(seq_preds))
    with open(output_file, "w") as f:
        for i, id in enumerate(test_ref_seqs):
            for loc in range(len(all_preds[i])):
                print(loc+1, id, seq_host[i], all_truths[i][loc], all_preds[i][loc], all_preds_prob[i][loc], sep="\t", file=f)
if __name__ == "__main__":
    main()
