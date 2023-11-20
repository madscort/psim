from pathlib import Path
import pytorch_lightning as pl
import pandas as pd
import torch
from Bio import SeqIO
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from src.data.get_sequence import get_gene_gc_sequence
import torch.nn.functional as F

def collate_fn_pad(batch, pad_length=56):
    sequences, labels = zip(*batch)

    for seq in sequences:
        if len(seq) > pad_length:
            raise ValueError(f"A sequence is longer than the maximum allowed length of {pad_length}")

    # Pad sequences that are shorter than pad_length
    sequences_padded = [F.pad(torch.tensor(seq), (0, max(0, pad_length - len(seq))), "constant", 0) for seq in sequences]
    sequences_padded = torch.stack(sequences_padded, dim=0)
    
    labels = torch.stack(labels)

    return sequences_padded, labels

def encode_sequence(sequence, vocabulary_mapping):
    return [vocabulary_mapping[gene_name] for gene_name in sequence]

translate = str.maketrans("ACGTURYKMSWBDHVN", "0123444444444444")

class SequenceDataset(Dataset):
    """
    Instantiate with path to sequence and label file
    """
    def __init__(self, input_sequences, input_labels, vocab_map=None):
        
        # Init is run once, when instantiating the dataset class.
        # It takes a list of labels and a list of paths to sequence files.
        self.labels = input_labels
        self.sequences_paths = input_sequences
        self.vocab_map = vocab_map

    def classes(self):
        
        # Returns the classes in the dataset (optional function)
        return self.labels.unique()

    def __len__(self):
        
        # Returns the number of samples in dataset (required)
        return len(self.labels)

    def __getitem__(self, idx): 
        # Returns a sample at position idx (required)
        # A sample includes:
        # - Sequence (one-hot encoded)
        # - Label (binary)

        # Load sequence. 
        with open(self.sequences_paths[idx], "r") as f:
            sequence = next(SeqIO.parse(f, "fasta")).seq

        # One-hot encode sequence
        sequence = torch.tensor([int(base.translate(translate)) for base in sequence], dtype=torch.float)
        sequence = one_hot(sequence.to(torch.int64), num_classes=5).to(torch.float).permute(1, 0)

        # Load label
        label = self.labels[idx]

        return sequence, label

class GCSequenceDataset(SequenceDataset):
    def __getitem__(self, idx):
        # Returns a sample at position idx
        # A sample includes:
        # - Sequence of GC contents (for each gene)
        # - Label (binary)

        # Get sequence of GC contents
        gc_contents = get_gene_gc_sequence(self.sequences_paths[idx]) 
        sequence = torch.tensor(gc_contents, dtype=torch.float).unsqueeze(-1)

        # Load label
        label = self.labels[idx]

        return sequence, label

class HMMMatchSequenceDataset(SequenceDataset):
    def __getitem__(self, idx):
        # Returns a sample at position idx
        # A sample includes:
        # - Sequence of HMM profile matches (for each gene)
        # - Label (binary)

        encoded_sequences = torch.tensor(encode_sequence(self.sequences_paths[idx], self.vocab_map))
        # Load label
        label = self.labels[idx]

        return encoded_sequences, label

class FixedLengthSequenceModule(pl.LightningDataModule):
    def __init__(self, return_type: str="fna", use_saved: bool=False, dataset: Path=None,
                 train_indices=None, val_indices=None, test_indices=None,
                 num_workers: int=1, batch_size: int=32, pad_pack: bool=False):
        super().__init__()
        self.return_type = return_type
        self.use_saved = use_saved
        self.vocab_map = None
        self.vocab_size = None
        self.max_seq_length = 0
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices
        self.dataset = dataset
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.string_model = False
        self.vocab_map = None
        
        if pad_pack:
            self.collate_fn = collate_fn_pad
        else:
            self.collate_fn = None

        match self.return_type:
            case "fna":
                self.DatasetType = SequenceDataset
            case "gc_sequence":
                self.DatasetType = GCSequenceDataset
            case "hmm_match_sequence":
                self.DatasetType = HMMMatchSequenceDataset
                self.string_model = True

    def prepare_data(self):
        pass

    def setup(self, stage: str = None):
        
        if self.string_model:
            self.data_splits = torch.load(self.dataset / "strings" / "allDB" / "dataset.pt")
            vocab = set()
            for split in ['train', 'val', 'test']:
                self.max_seq_length = max(self.max_seq_length, max(len(seq) for seq in self.data_splits[split]['sequences']))
                vocab.update(x for seq in self.data_splits[split]['sequences'] for x in seq)
            self.vocab_map = {name: i+1 for i, name in enumerate(sorted(vocab))}
            self.vocab_map['<PAD>'] = 0
            self.vocab_size = len(self.vocab_map)
        else:
            self.data_splits = {
                split: self.load_split_data(split) for split in ['train', 'val', 'test']
            }
        
        num_class_1 = self.data_splits['test']['labels'].sum()
        num_class_0 = len(self.data_splits['test']['labels']) - num_class_1
        class_counts = [num_class_0, num_class_1]  # replace with your actual class counts
        total = sum(class_counts)
        class_weights = [total / class_count for class_count in class_counts]
        self.class_weights = torch.tensor(class_weights, dtype=torch.float)
        self.steps_per_epoch = int(total) // self.batch_size

        if self.train_indices is not None:
            self.resplit_data()

    def load_split_data(self, split):
        df = pd.read_csv(self.dataset / f"{split}.tsv", sep="\t", header=0, names=["id", "type", "label"])
        sequences = [self.dataset / split / "sequences" / f"{id}.fna" for id in df['id'].values]
        labels = torch.tensor(df['label'].values)
        return {'sequences': sequences, 'labels': labels}

    def resplit_data(self):
        data_sequences = []
        data_labels = []
        for split in ['train', 'val', 'test']:
            data_sequences.extend(self.data_splits[split]['sequences'])
            data_labels.extend(self.data_splits[split]['labels'])

        self.data_splits = {
            'train': {'sequences': [data_sequences[i] for i in self.train_indices], 'labels': [data_labels[i] for i in self.train_indices]},
            'val': {'sequences': [data_sequences[i] for i in self.val_indices], 'labels': [data_labels[i] for i in self.val_indices]},
            'test': {'sequences': [data_sequences[i] for i in self.test_indices], 'labels': [data_labels[i] for i in self.test_indices]}
        }

    def train_dataloader(self):
        return DataLoader(self.DatasetType(self.data_splits['train']['sequences'], self.data_splits['train']['labels'], vocab_map=self.vocab_map), batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.collate_fn, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.DatasetType(self.data_splits['val']['sequences'], self.data_splits['val']['labels'], vocab_map=self.vocab_map), batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.DatasetType(self.data_splits['test']['sequences'], self.data_splits['test']['labels'], vocab_map=self.vocab_map), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.collate_fn, persistent_workers=True)
