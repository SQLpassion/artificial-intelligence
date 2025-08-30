from MultiHeadAttention import MultiHeadAttention
from LayerNormalization import LayerNormalization
from FeedForward import FeedForward
import torch.nn as nn
import torch

class TransformerBlock(nn.Module):
    # Constructor
    def __init__(self, cfg):
        super().__init__()

        # Initialize the individual modules
        self.attention = MultiHeadAttention(
            d_in = cfg["embedding_dimensions"],
            d_out = cfg["embedding_dimensions"],
            context_length = cfg["context_length"],
            num_heads = cfg["number_heads"],
            dropout = cfg["drop_rate"],
            qkv_bias = cfg["qkv_bias"]
        )
        self.feedforward = FeedForward(cfg)
        self.normalization1 = LayerNormalization(cfg["embedding_dimensions"])
        self.normalization2 = LayerNormalization(cfg["embedding_dimensions"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    # Forward Function
    def forward(self, x):
        # 1st block: Self Attention with Residual connection
        shortcut = x
        x = self.normalization1(x)
        x = self.attention(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # 2nd block: Feed Forward with Residiual connection
        shortcut = x
        x = self.normalization2(x)
        x = self.feedforward(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x