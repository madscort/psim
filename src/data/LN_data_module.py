from pathlib import Path
import pytorch_lightning as pl
import pandas as pd
import pyrodigal
import torch
from sklearn.model_selection import train_test_split
from Bio import SeqIO
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from src.data.get_sequence import get_gene_gc_sequence

# Collate function for non-LSTM
def collate_fn_zip(batch):
    sequences, labels = zip(*batch)
    labels = torch.tensor(labels)
    return sequences, labels

def collate_fn_pad_pack(batch):
    # Sort batch in descending order based on sequence length
    batch = sorted(batch, key=lambda x: x[0].size(0), reverse=True)
    
    # Get the sequences and labels separately
    sequences, labels = zip(*batch)
    
    # Get the length of each sequence
    lengths = [len(seq) for seq in sequences]
    
    # Pad the sequences
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    
    # Create packed sequence (if you plan to use this directly in your LSTM layer)
    sequences_packed = pack_padded_sequence(sequences_padded, lengths, batch_first=True)
    
    labels = torch.stack(labels)

    return sequences_packed, lengths, labels


translate = str.maketrans("ACGTURYKMSWBDHVN", "0123444444444444")

class SequenceDataset(Dataset):
    """
    Instantiate with path to sequence and label file
    """
    def __init__(self, input_sequences, input_labels):
        
        # Init is run once, when instantiating the dataset class.
        # It takes a list of labels and a list of paths to sequence files.
        self.labels = input_labels
        self.sequences_paths = input_sequences

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

class FixedLengthSequenceModule(pl.LightningDataModule):
    def __init__(self, return_type: str="fna",
                 dataset: Path=Path("data/processed/10_datasets/phage_25_fixed_25000"),
                 train_indices=None, val_indices=None, test_indices=None,
                 num_workers: int=1, batch_size: int=68, pad_pack: bool=False):
        super().__init__()
        self.return_type = return_type
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices
        self.datafolder = Path(dataset, "sequences")
        self.sampletable = Path(dataset, "sampletable.tsv")
        self.num_workers = num_workers
        self.batch_size = batch_size
        if pad_pack:
            self.collate_fn = collate_fn_pad_pack
        else:
            self.collate_fn = collate_fn_zip


        match self.return_type:
            case "fna":
                self.DatasetType = SequenceDataset
            case "gc_sequence":
                self.DatasetType = GCSequenceDataset
                

    def prepare_data(self):
        pass

    def setup(self, stage: str = None):
        
        # Load sampletable
        df_sampletable = pd.read_csv(self.sampletable, sep="\t", header=None, names=['id', 'type', 'label'])
        
        if self.train_indices is not None and self.val_indices is not None:
            # Used for cross-validation
            train = df_sampletable.iloc[self.train_indices]
            val = df_sampletable.iloc[self.val_indices]
            test = df_sampletable.iloc[self.test_indices]
        else:
            fit, test = train_test_split(df_sampletable, stratify=df_sampletable['type'], test_size=0.1)
            train, val = train_test_split(fit, stratify=fit['type'], test_size=0.2)
        
        self.train_sequences = [Path(self.datafolder, f"{id}.fna") for id in train['id'].values]
        self.val_sequences = [Path(self.datafolder, f"{id}.fna") for id in val['id'].values]
        self.test_sequences = [Path(self.datafolder, f"{id}.fna") for id in test['id'].values]
        
        self.train_labels = torch.tensor(train['label'].values)
        self.val_labels = torch.tensor(val['label'].values)
        self.test_labels = torch.tensor(test['label'].values)
        

    def train_dataloader(self):
        return DataLoader(self.DatasetType(self.train_sequences, self.train_labels), batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.DatasetType(self.val_sequences, self.val_labels), batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.DatasetType(self.test_sequences, self.test_labels), batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)
