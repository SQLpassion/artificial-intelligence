from CasualAttention import CasualAttention
import torch.nn as nn
import torch

class MultiHeadAttentionWrapper(nn.Module):
    # Constructor
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias = False):
        super().__init__()
        
        self.heads = nn.ModuleList([CasualAttention(d_in, d_out, context_length, dropout, qkv_bias)
                                    for _ in range(num_heads)])
        
    # Forward function
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim = -1)