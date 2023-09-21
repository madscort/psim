import torch
import wandb
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        kernel_size = (5,7)

        self.activation_fn = nn.ReLU()

        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        
        self.conv1 = nn.Conv1d(in_channels=5, out_channels=16,
                               kernel_size=kernel_size[0],
                               stride=1,
                               padding=padding[0])
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32,
                               kernel_size=kernel_size[1], stride=1, padding=padding[1])
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc_pre = nn.Linear(32, 32)
        self.fc = nn.Linear(32, 2)
        self.dropout = nn.Dropout(0.5)


    def forward(self, x):
        x = self.conv1(x)
        x = self.activation_fn(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.activation_fn(x)
        x = self.bn2(x)
        
        x = torch.mean(x, dim=2)
        
        x = self.fc_pre(x)
        x = self.activation_fn(x)
        x = self.dropout(x)

        x = self.fc(x)
        x = self.dropout(x)
        x = torch.softmax(x, dim=1)
        return x

class CaptumModule(pl.LightningModule):
    def __init__(self, lr=0.001, optimizer='adamw'):
        super(CaptumModule, self).__init__()
        
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.optimizer = optimizer
        self.model = CNN()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        _, loss, acc = self._get_preds_loss_accuracy(batch)
        self.log('train_loss', loss, batch_size=batch[1].shape[0])
        self.log('train_acc', acc, prog_bar=True, batch_size=batch[1].shape[0])
        return loss
    
    def validation_step(self, batch, batch_idx):
        preds, loss, acc = self._get_preds_loss_accuracy(batch)
        self.log('val_loss', loss, prog_bar=True, batch_size=batch[1].shape[0])
        self.log('val_acc', acc, prog_bar=True, batch_size=batch[1].shape[0])
        return preds
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = torch.softmax(self(x), dim=1)

        if not hasattr(self, 'test_outputs'):
            self.test_outputs = []
        self.test_outputs.append({'y_true': y, 'y_hat': y_hat})

        return {'y_true': y, 'y_hat': y_hat}

    def on_test_epoch_end(self):
        # concatenate all y_true and y_hat from outputs of test_step
        y_true = torch.cat([x['y_true'] for x in self.test_outputs], dim=0)
        y_hat = torch.cat([x['y_hat'] for x in self.test_outputs], dim=0)

        # convert probabilities to predicted labels
        y_pred = torch.argmax(y_hat, dim=1)

        y_true = y_true.cpu()
        y_pred = y_pred.cpu()
        y_hat = y_hat.cpu()
        
        # calculate metrics
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='binary')
        precision = precision_score(y_true, y_pred, average='binary')
        #auc = roc_auc_score(y_true, y_hat)
        
        # log metrics
        self.log('test_acc', torch.tensor(acc, dtype=torch.float32))
        self.log('test_f1', torch.tensor(f1, dtype=torch.float32))
        self.log('test_precision', torch.tensor(precision, dtype=torch.float32))
        #self.log('test_auc', torch.tensor(auc, dtype=torch.float32))
        
        # log ROC curve in wandb
        y_true_np = y_true.numpy()
        y_hat_np = y_hat.numpy()
        
        # self.logger.experiment.log({"roc": wandb.plot.roc_curve(y_true_np, y_hat_np, labels=["Class 0", "Class 1"], classes_to_plot=[1])})

        # self.logger.experiment.log({"performance": wandb.Table(columns=["accuracy", "f1", "precision", "auc"],
        #             data=[[acc, f1, precision, auc]])})

        del self.test_outputs
        return {'y_true': y_true, 'y_hat': y_hat}

    def configure_optimizers(self):
        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        elif self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        elif self.optimizer == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr)
        else:
            raise ValueError('Optimizer not supported')
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.95 ** epoch)
        monitor = 'val_loss'
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': monitor}
    
    def _get_preds_loss_accuracy(self, batch):
        '''convenience function since train/valid/test steps are similar'''
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        loss = self.criterion(logits, y)
        acc = accuracy(preds, y, 'binary')
        return preds, loss, acc
