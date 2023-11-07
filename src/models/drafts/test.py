import torch
from pathlib import Path
from src.data.attach_pangenome import get_one_string, get_hmms
from src.models.LNSequenceModule import SequenceModule
from src.data.LN_data_module import encode_sequence
from torch.nn.functional import softmax
import sys
sequence = """TTCTTCTTACAGGAGTTTTAGGTTACATACCATATAAATATCTAACAATGATAGGTTTAG
TTAGTGAAAAAAACAAGGTTATCAATACTCCTGTATTATTGATTTTTTCTATTGAAACAT
GTTTGATATGGTTTTATAGTTTTATAGTTTTTAATAATGTTGATTTAAAAAATTTGAATT
TAATTCAGTTGCTTACAGGTCTAAAAGCAAATATTTTGTTTCTATTTATTTTTGTTTTAA
CAGTGTTTGTATTTAATCCTTTAATTGTTAAATTTATTATCTGGTTAATTAATATAACCA
GAAAGTTTATGAAATTGGATTGTATAAGCTTATTAGACAAAAGAGACAAGTTGTTTAATA
ACAACGGTAAACCAGTATTTATAGTTATAAAAGACTTTGAAAACAGAATCATTGAAGAGG
GTGAACTTAAAACCTATAATTCAGCTGGTAGCGATTTCGATTTACTAGAAGTTGAGCGAC
AAGATTTCAAAGTATCTGATTTACCGTCAAACGATGAATTGTATATTAAACATACGCTTG
TAGACCTTAAACAACAAATTAAATTGGATTTATATTTAATGAATGAATACTAATCTTTTT
TCTTAGCTTTTTCTGATAAAGTGCTTTTTAATTTTTCGCTGGCGCCTGACTTTTCAAAAC
TTTTGTTTAATGGGTTACTACGAGTAGTTTCTTGTTTTTTGTTTTTATCTACCATAAAAT
TCTCACCACCATTCAACGTCTACACTAGTAGGCGTTTTTTGATTTTTATATTAAAGGGCT
ATAAAAAGCTGTTAATACTTCAATTCTTTAATCCACATATATTTAAAAGTGAGGTAGTAG
GTAATAAATATAAGACTTAAAGTTAAGATTGCTTTTTTCATGTCAATTTCTCCTTTGTTT
ATATTTATATTAAATCACTAAATAGACGTTATTAATCACAATACAATTAATTGATTGTAA
GATACTTAGTCGTATAATTCTATATACCTATTAGTAAATTCTTCTGCTGTTATTTCTCCA
TTTTCTTTTTGTTGTTGAAGTTTAGAAGCTTCTTTTTGAATTGCATCGTATTTTTCACGA
GAATACCCATATTTTTCCATCTCTTTATAATTAGCTTCGTTTATTTGTTCTTGTTGCTGA
GGTGTGACACAACCACCAACTGTGCATTGTGTACCATCAGGTTTTGTGTAACCTATAACG
TCACCTGCGCCTTGTGCTTGGTACCAAGTATTACCATCTGCATCTACCATGCCGTTAACA
TTGTGACCATTTTTTACTCTTTGTGATATTTCGTCTTTAGTTAAAGGTCTATTGGTTTGT
TGATCGTTGTTAACGTTTGTGTTGTTCTCGTTGTTTACTTGATTATTGTTATCGTTTTGA
TTAGCATTTTCTTTTTTCGCTTCTGCTTTTTCTTTAGTTTCTTTCTTTTTATCTTTGTTA
TCTTTCTTTGTTTCAGTTTTTTTGCTTTCCTCTTTCTTATCGCCGTCGTGGCTACCACAA
GCGCCTAAAACTAACGCACTCGCTAATGTTAAACCTAATAATCTTTTCATTTTAATTTCT
CCTTTGTTTATATTTCTTTATATTTAAAAACTCTCAATGGCTCAAATGTAATTGAGTATT
CGCCGTAGTGAGTCCCAATACCATATATCTTTTTATATTGTTCTATTGCTTCTAATATGT
ATTCTTCACTCAATTGCAGATACTCAGACAACTCATACAAGTTACGTACACCATAATTGT
AAGCTTCCACAATTTCGCGTAACGGGACTGCTGAGATAAAGCCGTGTCGCCTTGCGTAAT
TTTCGAACTTGCGATTGTTGAATTTCGAGTAATCGGCTATATCACCGTATGTAAGTTTAT
TATGTGCTAATTCTTCAAAGAGAATTCCTGCCTTTTCTCTATCTGATAAGCCACGCTTTA
TTAAAATTAAATCTCCTAACCATACCCCATCCAAATTATCTGGAAGCACATCAGCCTCTC"""

vocab_map = torch.load(Path("data/processed/10_datasets/dataset_v01/strings/pfama/vocab_map.pt"))
data_splits = torch.load(Path("data/processed/10_datasets/dataset_v01/strings/pfama/dataset.pt"))
model = SequenceModule.load_from_checkpoint(checkpoint_path=Path("psim/wahn6i9x/checkpoints/epoch=8-step=4806.ckpt").absolute(),
                                                map_location=torch.device('cpu'))
model.eval()

for split in ['test']:
    for n in range(len(data_splits['test']['labels'])):
        input_string = data_splits[split]['sequences'][n]
        encoded_string = torch.tensor(encode_sequence(input_string, vocab_map))
        print(encoded_string)
        sys.exit()
        #prediction = model({"seqs": encoded_string.unsqueeze(0)})
        #print(prediction.argmax(dim=1).tolist()[0])



sys.exit()
sequence = sequence.replace("\n", "")
model_string = get_one_string(sequence, Path("data/processed/10_datasets/attachings/hmms/hmms.hmm"))
encoded_string = torch.tensor(encode_sequence(model_string, vocab_map))
res = model({"seqs": encoded_string.unsqueeze(0)})

# print(res)