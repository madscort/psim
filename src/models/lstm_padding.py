import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

# Step 1: Data Preparation
sequences = [
    torch.tensor([[1], [2], [3], [4]]),  # ACGT
    torch.tensor([[2], [3], [1]]),       # CGA
    torch.tensor([[4], [1], [3], [2], [3]]) # TGCGC
]


# Step 2: Padding
sequences_sorted = sorted(sequences, key=len, reverse=True)
sequences_padded = pad_sequence(sequences_sorted, batch_first=True)

# Getting the lengths of each sequence
lengths = [len(seq) for seq in sequences_sorted]

# Step 3: Packing
packed_sequences = pack_padded_sequence(sequences_padded, lengths, batch_first=True)

# Define a simple LSTM model
class SimpleLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, output_size=1):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, lengths):
        x_packed = pack_padded_sequence(x, lengths, batch_first=True)
        packed_output, (hn, cn) = self.lstm(x_packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        idx = (torch.tensor(lengths) - 1).view(-1, 1).expand(len(lengths), output.size(2)).unsqueeze(1)
        output = output.gather(1, idx).squeeze(1)
        output = self.fc(output)
        
        return output

# Initializing and calling the model
model = SimpleLSTM()
output = model(sequences_padded.float(), lengths)
