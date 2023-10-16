import torch
import torch.nn as nn

class BasicTransformer(nn.Module):
    def __init__(self,
                num_classes: int,
                fc_dropout_rate: float,
                vocab_size: int,
                embedding_dim: int,
                num_heads: int,
                num_layers: int,
                dim_feedforward: int,
                dim_fc: int):
        super(BasicTransformer, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.classifier_head = nn.Sequential(
            nn.Linear(embedding_dim, dim_fc),
            nn.ReLU(),
            nn.Dropout(fc_dropout_rate),
            nn.Linear(dim_fc, num_classes),
            nn.Dropout(fc_dropout_rate),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = x["seqs"]
        padding_mask = self.generate_padding_mask(x).t()
        x = self.embedding(x)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        x = torch.mean(x, dim=1)
        x = self.classifier_head(x)
        return x
    
    def generate_padding_mask(self, x, pad_value=0):
        return (x == pad_value)