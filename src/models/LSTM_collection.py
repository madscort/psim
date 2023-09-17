import torch
import torch.nn as nn

class BasicLSTM(nn.Module):
    def __init__(self,
                alt_dropout_rate: float=0.1,
                fc_dropout_rate: float=0.5,
                activation_fn: str='ReLU',
                batchnorm: bool=True,
                fc_num: int=1,
                kernel_size: tuple=(3,3,3),
                num_inception_layers: int = 5,
                out_channels: int = 16,
                kernel_size_b1: int = 3,
                kernel_size_b2: int = 5,
                keep_b3 = True,
                keep_b4 = True,
                input_size=25000,
                hidden_size_lstm=64,
                num_layers_lstm=1,
                num_classes=1):
        super(BasicLSTM, self).__init__()
        
        self.fc_num = fc_num
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size_lstm, 
                            num_layers=num_layers_lstm, 
                            batch_first=True, 
                            bidirectional=True)
        
        self.fc_pre = nn.Linear(2 * hidden_size_lstm, 2 * hidden_size_lstm)
        self.fc = nn.Linear(2 * hidden_size_lstm, num_classes)

        self.dropout = nn.Dropout(fc_dropout_rate)
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = torch.mean(x, dim=1)
        if self.fc_num == 2:
            x = self.fc_pre(x)
            x = self.dropout(x)
        x = self.fc(x)
        x = self.dropout(x)
        
        return x.squeeze(-1)