import torch
import torch.nn as nn
import math

class BasicTransformer(nn.Module):
    def __init__(self,
                num_classes: int,
                fc_dropout_rate: float,
                vocab_size: int,
                max_seq_length: int,
                embedding_dim: int,
                num_heads: int,
                num_layers: int,
                dim_feedforward: int,
                dim_fc: int):
        super(BasicTransformer, self).__init__()
        max_seq_length = 56
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, fc_dropout_rate, max_seq_length)
        encoder_layer = nn.TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.classifier_head = nn.Sequential(
            nn.Linear(embedding_dim, dim_fc),
            nn.ReLU(),
            nn.Dropout(fc_dropout_rate),
            nn.Linear(dim_fc, num_classes),
            nn.Dropout(fc_dropout_rate)
        )
        
    def forward(self, x):
        padding_mask = self.generate_padding_mask(x)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        x = torch.mean(x, dim=1)
        x = self.classifier_head(x)
        return x
    
    def generate_padding_mask(self, x, pad_value=0):
        return (x == pad_value).bool()
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x dimensions: [batch_size, seq_len, embedding_dim]
        # pe dimensions: [max_len, 1, embedding_dim]
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)