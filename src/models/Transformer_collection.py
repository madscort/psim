import torch
import torch.nn as nn

class BasicTransformer(nn.Module):
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
                num_classes=2,
                pad_pack: bool=False,
                embedding_dim=None,
                vocab_size=5,
                # Transformer only:
                d_model=10,
                nhead=5,
                num_layers=1,
                dim_feedforward=10):
        super(BasicTransformer, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.classifier_head = nn.Sequential(
            nn.Linear(d_model, 10),
            nn.ReLU(),
            nn.Linear(10, num_classes),
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