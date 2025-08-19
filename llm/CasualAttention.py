import torch.nn as nn
import torch

class CasualAttention(nn.Module):
    # Constructor
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias = False):
        super().__init__()
        
        # Initialize the trainable weight matrices
        self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.d_out = d_out

        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal = 1))

    # Forward function
    def forward(self, x):
        # Calculate the weights
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Calculate the attention weights and the context vector
        attention_scores = queries @ keys.transpose(1, 2)
        attention_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attention_weights = torch.softmax(attention_scores / keys.shape[-1] ** 0.5, dim = -1)
        attention_weights = self.dropout(attention_weights)

        context_vector = attention_weights @ values

        # Return the computed context vector
        return context_vector