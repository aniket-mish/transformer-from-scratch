import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model

        # dictionary kind of layer that maps the input to a vector of dimension d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, input):
        return self.embedding(input) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        # Create a tensor of shape (max_len, 1).
        position = torch.arange(0, max_len).unsqueeze(1)
        # Calculating log before taking exponential is more numerically stable.
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # Taking sin for even indices.
        pe[:, 0::2] = torch.sin(position * div_term)
        # Taking cos for odd indices.
        pe[:, 1::2] = torch.cos(position * div_term)
        # Add a batch dimension (1, max_len, d_model).
        pe = pe.unsqueeze(0)

        # Lets you save and restore tensors that are not model parameters but are still part of the model state.
        # For example, a buffer that should not be updated during backpropagation using gradient descent but should be a part of the model's state.
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        # For numerical stability and to avoid division by zero
        self.eps = eps
        # These are learnable parameters(nn.Parameter)
        self.alpha = nn.Parameter(torch.ones(1)) # multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # added

    def forward(self, x):
        # dim=-1 => to calculate mean on the last dimension.
        # keepdim=True => generally mean cancels the dimension it is applied on, but we want to keep it.
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # W1 and b1
        self.w_1 = nn.Linear(d_model, d_ff) # bias=True by default
        # W2 and b2
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # w_1 converts the shape of x from (batch_size, max_len, d_model) to (batch_size, max_len, d_ff)
        # w_2 converts the shape of x from (batch_size, max_len, d_ff) to (batch_size, max_len, d_model)
        return self.w_2(self.dropout(torch.relu(self.w_1(x))))


class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_model, h, dropout):
        super().__init__()

        # Make sure d_model is always divisible by h
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h

        self.w_q = nn.Linear(d_model, d_model) # Wq
        self.w_k = nn.Linear(d_model, d_model) # Wk
        self.w_v = nn.Linear(d_model, d_model) # Wv

        self.w_o = nn.Linear(d_model, d_model) # Wo
        self.dropout = nn.Dropout(dropout)


    # @staticmethod
    # def attention(query, key, value, mask=None, dropout=None):


    def forward(self, q, k, v, mask=None):
        query = self.w_q(q) # (batch_size, max_len, d_model) => (batch_size, max_len, d_model)
        key = self.w_k(k)
        value = self.w_v(v)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # (batch_size, max_len, d_model) => (batch_size, max_len, h, d_model) => (batch_size, h, max_len, d_k)
        query = query.view(query.size(0), -1, self.h, self.d_k).transpose(1, 2)
        key = key.view(key.size(0), -1, self.h, self.d_k).transpose(1, 2)
        value = value.view(value.size(0), -1, self.h, self.d_k).transpose(1, 2)


