import torch.nn as nn
import torch

class MultiHeadAttention(nn.Module):
    # Constructor
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias = False):
        super().__init__()

        self.d_out = d_out                                      # Number of Output Dimensions
        self.num_heads = num_heads                              # Number of Heads
        self.head_dim = d_out // num_heads                      # Dimensions per Head
        self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.out_projection = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal = 1))

    # Forward function
    def forward(self, x):
        # Generates a Tensor with the shape "(b, num_tokens, d_out)"
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Split "d_out" across the heads: d_out = num_heads * head_dim
        # Generates a Tensor with the shape "(b, num_tokens, num_heads, head_dim)"
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose from the shape "(b, num_tokens, num_heads, head_dim)"" to the shape
        # "(b, num_heads, num_tokens, head_dim)".
        # The dimensions "num_tokens" and "num_heads" are just swapped.
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        queries = queries.transpose(1, 2)

        # Compute the attention scores for each head
        attention_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attention_scores.masked_fill_(mask_bool, -torch.inf)

        # Compute the attention weights
        attention_weights = torch.softmax(attention_scores / keys.shape[-1] ** 0.5, dim = -1)
        attention_weights = self.dropout(attention_weights)

        # Compute the context vector and merge the individual heads together
        context_vector = (attention_weights @ values).transpose(1, 2)
        context_vector = context_vector.contiguous().view(b, num_tokens, self.d_out)
        context_vector = self.out_projection(context_vector)

        return context_vector