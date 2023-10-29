import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, input):
        return self.embedding(input) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, seq_len: int):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.seq_len = seq_len # max len of the sentences
        self.register_buffer('positional_encoding', self._get_positional_encoding()) # lets you save and restore tensors that are not model parameters but are still part of the model state.

    # create a tensor of shape (seq_len, d_model)
    def _get_positional_encoding(self):
        pe = torch.zeros(self.seq_len, self.d_model)
        # create a tensor of shape (seq_len, 1)
        position = torch.arange(0, self.seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
        # sin for even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # cos for odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        # add a batch dimension (1, seq_len, d_model)
        return pe.unsqueeze(0)

    def forward(self, x):
        x = x + self.positional_encoding[:, :x.size(1)].requires_grad_(False) # shape and size returns the same thing
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # learnable parameter
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True) # -1 means the last dimension
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias