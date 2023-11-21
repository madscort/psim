import torch
import torch.nn as nn
import math

import torch
import torch.nn as nn
import math
import sys

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
        # print("Input: ", x.shape)
        # print(x)
        padding_mask = self.generate_padding_mask(x)
        # print("Padding mask: ", padding_mask.shape)
        # print(padding_mask)
        x = self.embedding(x)
        # print("After embedding: ", x.shape)
        x = self.pos_encoder(x)
        # print("After pos encoding: ", x.shape)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        # print("After transformer: ", x.shape)
        x = torch.mean(x, dim=1)
        # print("After mean: ", x.shape)
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
        # x is [batch_size, seq_len, embedding_dim]
        # 'pe' is [max_len, 1, embedding_dim]
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

def test_model():
    # Model parameters - adjust as needed
    num_classes = 10
    fc_dropout_rate = 0.1
    vocab_size = 1000
    max_seq_length = 50
    embedding_dim = 128
    num_heads = 4
    num_layers = 2
    dim_feedforward = 512
    dim_fc = 256

    # Create the model instance
    model = BasicTransformer(
        num_classes=num_classes,
        fc_dropout_rate=fc_dropout_rate,
        vocab_size=vocab_size,
        max_seq_length=max_seq_length,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dim_fc=dim_fc
    )

    # Create a mock input tensor
    batch_size = 2
    seq_len = 30
    mock_input = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len))

    # Pass the mock input through the model
    output = model(mock_input)

    # Check output shape
    expected_output_shape = (batch_size, num_classes)
    assert output.shape == expected_output_shape, f"Output shape is incorrect. Expected: {expected_output_shape}, Got: {output.shape}"

    print("Output shape is correct.")

    # Check for NaN values in the output
    assert not torch.isnan(output).any(), "Model output contains NaN values."

    print("No NaN values in output.")


def test_positional_encoding():
    # Parameters for the model
    num_classes = 10
    fc_dropout_rate = 0.1
    vocab_size = 1000
    max_seq_length = 50
    embedding_dim = 128
    num_heads = 4
    num_layers = 2
    dim_feedforward = 512
    dim_fc = 256

    # Initialize the model
    model = BasicTransformer(
        num_classes=num_classes,
        fc_dropout_rate=fc_dropout_rate,
        vocab_size=vocab_size,
        max_seq_length=max_seq_length,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dim_fc=dim_fc
    )

    # Create a mock input tensor
    batch_size = 2
    seq_len = 30
    mock_input = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len))

    # Extract the positional encoder from the model
    pos_encoder = model.pos_encoder

    # Apply the positional encoding to the mock input
    encoded_input = model.embedding(mock_input)
    encoded_with_pos = pos_encoder(encoded_input)

    # Check if positional encoding changes the input
    assert not torch.equal(encoded_input, encoded_with_pos), "Positional encoding has no effect."

    print("Positional encoding alters the input.")

    # Ensure positional encoding doesn't change the shape of the input
    assert encoded_with_pos.shape == encoded_input.shape, "Positional encoding changes the input shape."

    print("Positional encoding retains the input shape.")

    # Check for NaN values after positional encoding
    assert not torch.isnan(encoded_with_pos).any(), "Positional encoding introduces NaN values."

    print("No NaN values after positional encoding.")

if __name__ == "__main__":
    test_model()

    test_positional_encoding()