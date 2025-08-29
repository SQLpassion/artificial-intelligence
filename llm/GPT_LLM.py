from DummyGPTModel import DummyGPTModel
from LayerNormalization import LayerNormalization
import torch.nn as nn
import tiktoken
import torch

# GPT Configuration data
GPT_CONFIG_124M = {
    "vocabulary_size": 50257,
    "context_length": 1024,
    "embedding_dimensions": 768,
    "number_heads": 12,
    "number_layers": 12,
    "drop_rate": 0.10,
    "qkv_bias": False
}

# Initialize the Tokenizer
tokenizer = tiktoken.get_encoding("gpt2")
batch = []
text1 = "Every effort moves you"
text2 = "Every day holds a"

# Prepare some tokens as input
batch.append(torch.tensor(tokenizer.encode(text1)))
batch.append(torch.tensor(tokenizer.encode(text2)))
batch = torch.stack(batch, dim = 0)
print(batch)

# Initialize and use the GPT model
torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)

# Output shape: torch.Size([2, 4, 50257])
#   => 2 rows, because we have 2 text samples
#   => Each text sample consists of 4 tokens
#   => Each token has a 50257-dimensional vector, which is the size of the vocabulary
print("Output shape: ", logits.shape)
print(logits)
print("")

######################
# Layer Normalization
######################
print("####################")
print("Layer Normalization")
print("####################")
torch.manual_seed(123)
batch_example = torch.randn(2, 5)
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
output = layer(batch_example)
print(output)
mean = output.mean(dim = -1, keepdim = True)
variance = output.var(dim = -1, keepdim = True)
print("Mean:", mean)
print("Variance: ", variance)
print("")

# Output normalized
torch.set_printoptions(sci_mode = False)
output_normalization = (output - mean) / torch.sqrt(variance)
mean = output_normalization.mean(dim = -1, keepdim = True)
variance = output_normalization.var(dim = -1, keepdim = True)
print("Mean - Normalized:", mean)
print("Variance - Normalized: ", variance)
print("")

# LayerNormalized module usage
ln = LayerNormalization(embedding_dimensions = 5)
out_ln = ln(batch_example)
mean = out_ln.mean(dim = -1, keepdim = True)
variance = out_ln.var(dim = -1, unbiased = False, keepdim = True)
print("Mean:", mean)
print("Variance: ", variance)