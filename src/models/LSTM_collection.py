import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class BasicLSTM(nn.Module):
    def __init__(self,
                fc_dropout_rate: float,
                activation_fn: str,
                fc_num: int,
                input_size: int,
                hidden_size_lstm: int,
                num_layers_lstm: int,
                num_classes: int,
                pad_pack: bool,
                embedding_dim: int,
                vocab_size: int,
                dim_shift: bool):
        super(BasicLSTM, self).__init__()
        self.dim_shift = dim_shift
        self.pad_pack = pad_pack
        self.fc_num = fc_num
        self.embedding_dim = embedding_dim
        if self.embedding_dim != 0:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            input_size = embedding_dim
        self.activation_fn = getattr(nn, activation_fn)()

        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size_lstm, 
                            num_layers=num_layers_lstm, 
                            batch_first=True, 
                            bidirectional=True)
        
        self.fc_pre = nn.Linear(2 * hidden_size_lstm, 2 * hidden_size_lstm)
        self.fc = nn.Linear(2 * hidden_size_lstm, num_classes)

        self.dropout = nn.Dropout(fc_dropout_rate)
        
    def forward(self, x):
        if self.pad_pack:
            lengths = x["lengths"]
            x = x["seqs"]
            if self.dim_shift:
                x = x.permute(0, 2, 1)
            if self.embedding_dim != 0:
                x = self.embedding(x)
            x = pack_padded_sequence(x, lengths, batch_first=True)
            x, _ = self.lstm(x)
            x, _ = pad_packed_sequence(x, batch_first=True)
        else:
            if self.dim_shift:
                x = x.permute(0, 2, 1)
            x, _ = self.lstm(x)
        x = torch.mean(x, dim=1)
        if self.fc_num == 2:
            x = self.fc_pre(x)
            x = self.activation_fn(x)
            x = self.dropout(x)
        x = self.fc(x)
        x = self.dropout(x)
        
        return x.squeeze(-1)
