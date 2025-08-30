from LayerNormalization import LayerNormalization
from TransformerBlock import TransformerBlock
from DummyGPTModel import DummyGPTModel
from FeedForward import FeedForward
from GPTModel import GPTModel
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
print("")

#######################
# Feed Forward Network
#######################
print("#####################")
print("Feed Forward Network")
print("#####################")
ffn = FeedForward(GPT_CONFIG_124M)
print(ffn)
x = torch.rand(2, 3, 768)
output = ffn(x)
print(output.shape)
print("")

####################
# Transformer Block
####################
print("##################")
print("Transformer Block")
print("##################")
torch.manual_seed(123)
x = torch.rand(2, 4, 768)
block = TransformerBlock(GPT_CONFIG_124M)
print(block)
output = block(x)
print("Input shape: ", x.shape)
print("Output shape: ", output.shape)
print("")

############
# GPT Model
############
print("##########")
print("GPT Model")
print("##########")

# A simple text generation loop
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        
        # Deactivates the gradient tracking, which is not needed during the generation phase
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        probabilities = torch.softmax(logits, dim = -1)
        idx_next = torch.argmax(probabilities, dim = -1, keepdim = True)
        idx = torch.cat((idx, idx_next), dim = 1)

    return idx

# Prepare some tokens as input
tokenizer = tiktoken.get_encoding("gpt2")
batch = []
text1 = "Every effort moves you"
text2 = "Every day holds a"
batch.append(torch.tensor(tokenizer.encode(text1)))
batch.append(torch.tensor(tokenizer.encode(text2)))
batch = torch.stack(batch, dim = 0)

# Run the GPT model
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
print(model)
output = model(batch)
print("Input Batch: ", batch)
print("Output Shape: ", output.shape)
print(output)

# Number of model parameter tensors
total_parameters = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_parameters:,}".replace(",", "."))
print("")

# Use the Text generation loop
start_context = "Hello, I am"
encoded_tokens = tokenizer.encode(start_context)
print("Encoded Tokens: ", encoded_tokens)
encoded_tensor = torch.tensor(encoded_tokens).unsqueeze(0)
print("Encoded Tensor shape: ", encoded_tensor.shape)

# Generate new text
model.eval()
output = generate_text_simple(model = model, 
                              idx = encoded_tensor,
                              max_new_tokens = 6,
                              context_size = GPT_CONFIG_124M["context_length"])
print("Output: ", output)
print("Output length: ", len(output[0]))

# Decode the generated text
decoded_text = tokenizer.decode(output.squeeze(0).tolist())
print(decoded_text)