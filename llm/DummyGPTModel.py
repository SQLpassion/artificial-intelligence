import torch.nn as nn
import torch

class DummyGPTModel(nn.Module):
    # Constructor
    def __init__(self, cfg):
        super().__init__()
        self.token_embeddings = nn.Embedding(cfg["vocabulary_size"], cfg["embedding_dimensions"])
        self.positional_embeddings = nn.Embedding(cfg["context_length"], cfg["embedding_dimensions"])
        self.drop_embeddings = nn.Dropout(cfg["drop_rate"])
        self.transformer_blocks = nn.Sequential(*[DummyTransformerBlock(cfg)
                                                  for _ in range(cfg["number_layers"])])
        self.final_normalization = DummyLayerNormalization(cfg["embedding_dimensions"])
        self.output_head = nn.Linear(cfg["embedding_dimensions"], cfg["vocabulary_size"], bias = False)

    # Forward function
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        token_embeddings = self.token_embeddings(in_idx)
        positional_embeddings = self.positional_embeddings(torch.arange(seq_len, device = in_idx.device))
        x = token_embeddings + positional_embeddings
        x = self.drop_embeddings(x)
        x = self.transformer_blocks(x)
        x = self.final_normalization(x)
        logits = self.output_head(x)

        return logits
    
class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return x

class DummyLayerNormalization(nn.Module):
    def __init__(self, normalized_shape, eps = 1e-5):
        super().__init__()

    def forward(self, x):
        return x