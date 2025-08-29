from GELU import GELU
import torch.nn as nn
import torch

class FeedForward(nn.Module):
    # Constructor
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["embedding_dimensions"], 4 * cfg["embedding_dimensions"]),
            GELU(),
            nn.Linear(4 * cfg["embedding_dimensions"], cfg["embedding_dimensions"]),
        )
    
    # Forward Function
    def forward(self, x):
        return self.layers(x)