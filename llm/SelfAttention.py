import torch.nn as nn
import torch

class SelfAttention(nn.Module):
    # Constructor
    def __init__(self, d_in, d_out, qkv_bias = False):
        super().__init__()

        # Initialize the trainable weight matrices
        self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias = qkv_bias)

    # Forward function
    def forward(self, x):
        # Calculate the weights
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Calculate the attention weights and the context vector
        attention_scores = queries @ keys.T
        attention_weights = torch.softmax(attention_scores / keys.shape[-1] ** 0.5, dim = -1)
        context_vector = attention_weights @ values

        # Return the computed context vector
        return context_vector