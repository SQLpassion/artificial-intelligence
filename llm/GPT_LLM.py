from DummyGPTModel import DummyGPTModel
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