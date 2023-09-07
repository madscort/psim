import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.functional import accuracy

class SequenceCNN(pl.LightningModule):
    def __init__(self, lr: float=0.001):
        super(SequenceCNN, self).__init__()
        self.save_hyperparameters()
        
        self.conv1 = nn.Conv1d(5, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.fc = nn.Linear(128, 1)
        self.relu = nn.ReLU()

        self.criterion = nn.BCEWithLogitsLoss()
        self.lr = lr

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = torch.mean(x, dim=2)  # Global average pooling
        x = self.fc(x)
        return x.squeeze(-1)

    def training_step(self, batch, batch_idx):
        _, loss, acc = self._get_preds_loss_accuracy(batch)
        self.log('train_loss', loss)
        self.log('train_acc', acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        preds, loss, acc = self._get_preds_loss_accuracy(batch)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return preds

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def _get_preds_loss_accuracy(self, batch):
        '''convenience function since train/valid/test steps are similar'''
        x, y = batch
        logits = self(x)
        preds = (torch.sigmoid(logits) > 0.5).long()
        loss = self.criterion(logits, y.float())
        acc = accuracy(preds, y, 'binary')
        return preds, loss, acc
