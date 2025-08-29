import torch.nn as nn
import torch

class LayerNormalization(nn.Module):
    # Constructor
    def __init__(self, embedding_dimensions):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(embedding_dimensions))
        self.shift = nn.Parameter(torch.zeros(embedding_dimensions))

    # Forward Function
    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        variance = x.var(dim = -1, keepdim = True, unbiased = False)
        normalized_x = (x - mean) / torch.sqrt(variance + self.eps)

        return self.scale * normalized_x + self.shift