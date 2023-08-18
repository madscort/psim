from io import BytesIO
import pytorch_lightning as pl
import torch
import os
from PIL import Image
from glob import glob
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

class SequenceDataset(Dataset):
    """
    Instantiate with path to sequence and label file
    """
    def __init__(self, input_sequences, input_labels):
        
        # Init is run once, when instantiating the dataset class.

        with open(input_labels, 'r') as file:
            self.labels = [int(line.strip()) for line in file]
        
        self.sequences_file = input_sequences

    def classes(self):
        
        # Returns the classes in the dataset (optional function)
        return self.labels

    def __len__(self):
        
        # Returns the number of samples in dataset (required)
        return len(self.labels)

    def __getitem__(self, idx):

        # Returns a sample at position idx (required)
        # A sample includes:
        # - Sequence (one-hot encoded)
        # - Label
        
        input_ids = self.sequences['input_ids'][idx]
        attention_mask = self.sequences['attention_mask'][idx]
        label = torch.tensor(self.labels[idx])
        num_labels = len(main2label.values()) if self.label_type_cat else len(class2label.values())
        
        sample = dict()
        sample['input_ids'] = input_ids
        sample['attention_mask'] = attention_mask
        sample['label'] = one_hot(label,
                                  num_classes = num_labels).to(torch.float)
        return sample


class StampDataset(Dataset):
    def __init__(self, predict_data):
        if os.path.isdir(predict_data):
            predict_data = glob(predict_data + "//**.jpg") + glob(
                predict_data + "//**.png"
            )
            self.predict_images = torch.stack(
                [transform(Image.open(x)) for x in predict_data]
            )
        else:
            if os.path.isfile(predict_data):
                img = Image.open(predict_data)
            else:
                img = Image.open(BytesIO(predict_data))
            resized_tensor = transform(img)
            self.predict_images = torch.unsqueeze(resized_tensor, dim=0)

    def __len__(self):
        return self.predict_images.size()[0]

    def __getitem__(self, idx):
        return self.predict_images[idx].float()


class StampDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./data/raw", predict_data=None):
        super().__init__()
        self.data_dir = data_dir
        self.predict_data = predict_data

    def prepare_data(self):
        # Download only if not prediction
        if self.predict_data is not None:
            return
        datasets.Flowers102(self.data_dir, "train", download=True)
        datasets.Flowers102(self.data_dir, "test", download=True)

    def setup(self, stage: str):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            dataset_full = datasets.Flowers102(
                self.data_dir, "train", transform=transform
            )
            self.dataset_train, self.dataset_val = random_split(
                dataset_full, [0.9, 0.1]
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.dataset_test = datasets.Flowers102(
                self.data_dir, "test", transform=transform
            )

        # Assign predict dataset from custom class and folder.
        if stage == "predict":
            self.dataset_predict = predictImageDataset(self.predict_data)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=68, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=68)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=68)

    def predict_dataloader(self):
        return DataLoader(self.dataset_predict, batch_size=68)
