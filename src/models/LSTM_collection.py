import torch
import timm
import wandb
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score, roc_curve



class BasicLSTM(nn.Module):
    def __init__(self, input_size=25000, hidden_size=64,
                 num_layers=1, num_classes=1, fc_dropout_rate: float = 0.50):
        super(BasicLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True, 
                            bidirectional=True)
        
        self.fc = nn.Linear(2 * hidden_size, num_classes)

        self.dropout = nn.Dropout(fc_dropout_rate)
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = torch.mean(x, dim=1)
        x = self.fc(x)
        x = self.dropout(x)
        
        return x.squeeze(-1)