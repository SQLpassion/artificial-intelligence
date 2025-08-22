from MultiHeadAttentionWrapper import MultiHeadAttentionWrapper
from MultiHeadAttention import MultiHeadAttention
from SelfAttention import SelfAttention
from CasualAttention import CasualAttention
from torch.utils.data import DataLoader
from torchviz import make_dot
from GPTDataset import GPTDataset
from pathlib import Path
import tiktoken
import torch

# Creates a DataLoader that wraps the Dataset
def create_dataloader(text,
                        batch_size = 4,
                        max_length = 256,
                        stride=128,
                        shuffle = True,
                        drop_last = True,
                        num_workers = 0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(text, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset,
                            batch_size = batch_size,
                            shuffle = shuffle,
                            drop_last = drop_last,
                            num_workers = num_workers)
    
    return dataloader

# Read a sample text from the file system
script_dir = Path(__file__).resolve().parent
file_path = script_dir / "The_Verdict.txt"

with file_path.open("r", encoding="utf-8") as file:
     raw_text = file.read() 

# Tokenize the whole text with the tiktoken tokenizer
tokenizer = tiktoken.get_encoding("gpt2")
encoded_text = tokenizer.encode(raw_text)
print("Number of total tokens: ", len(encoded_text))

# Remove the first 50 tokens from the dataset
encoded_sample = encoded_text[50:]
print(encoded_sample)
print("")

# Create Input-Target pairs:
# => The variable "x" contains the input tokens
# => The variable "y" contains the targets, which are the inputs shifted by 1
context_size = 4
x = encoded_sample[:context_size]
y = encoded_sample[1:context_size + 1]
print(f"x: {x}")
print(f"y:      {y}")
print("")

# We loop in a sliding window over the context size
for i in range(1, context_size + 1):
     context = encoded_sample[:i]
     desired = encoded_sample[i]
     print(context, "---->", desired)
     print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))
     print("")

# Create a DataLoader that wraps the whole raw text in a Dataset internally.
# The 1st tensor in the output stores the input token ids, and the 2nd tensor
# stores the target token ids.
# The 2nd batch token ids are shifted by one position:
# Input Tensor:  [40,  367, 2885, 1464]
# Output Tensor: [     367, 2885, 1464, 1807]
print("Working with a DataLoader")
dataloader = create_dataloader(raw_text, batch_size = 1, max_length = 4, stride = 1, shuffle = False)
data_iterator = iter(dataloader)
first_batch = next(data_iterator)
print("First batch:")
print(first_batch)
print("Input token ids: ", first_batch[0])
print("Output token ids:      ", first_batch[1])
print("")

# Fetch the next batch
print("Next batch:")
second_batch = next(data_iterator)
print(second_batch)
print("Input token ids: ", second_batch[0])
print("Output token ids:      ", second_batch[1])
print("")

# Working with Embeddings
vocabulary_size = 6
output_dimensions = 3
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocabulary_size, output_dimensions)
print(embedding_layer.weight)
input_ids = torch.tensor([2, 3, 5, 1])
print(embedding_layer(torch.tensor([3])))
print(embedding_layer(input_ids))
print("")

# Embed each token into a 256-dimensional vector
vocabulary_size = 50257
output_dimensions = 256
context_length = 4

dataloader = create_dataloader(raw_text, batch_size = 8, max_length = context_length, stride = 1, shuffle = False)
data_iterator = iter(dataloader)
inputs, targets = next(data_iterator)
print("Input Token IDs:\n", inputs)
print("Output Token IDs:\n", targets)
print("Input Shape: ", inputs.shape)

token_embedding_layer = torch.nn.Embedding(vocabulary_size, output_dimensions)
token_embeddings = token_embedding_layer(inputs) # Lookup into the embedding_layer based on the token IDs
print("token_embeddings: ", token_embeddings.shape)

# Create the absolute positioning embedding vectors
pos_embedding_layer = torch.nn.Embedding(context_length, output_dimensions)
pos_embeddings = pos_embedding_layer(torch.arange(context_length)) # Lookup into the pos_embedding_layer based on the positional values (0, 1, 3, 4, ...)
print("pos_embeddings: ", pos_embeddings.shape)

input_embeddings = token_embeddings + pos_embeddings
print("input_embeddings: ", input_embeddings.shape)
print("")

###################################
# Part 2: Self-Attention mechanism
###################################

#############################################
# Compute Attention Weights for just 1 input
#############################################

# Some dummy embedding values
inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89], # Your      (x^1)
        [0.55, 0.87, 0.66], # journey   (x^2)
        [0.57, 0.85, 0.64], # starts    (x^3)
        [0.22, 0.58, 0.33], # with      (x^4)
        [0.77, 0.25, 0.10], # one       (x^5)
        [0.05, 0.80, 0.55]  # step      (x^6)
    ]
)

# Example dot product:
# Attention Score between the current token (x^2) and the current input token (x^1)
#
# x^2  * x^1
# ====================
# 0.55 * 0.43 = 0.2365
# 0.87 * 0.15 = 0.1305
# 0.66 * 0.89 = 0.5874
# ====================
#               0.9544

# Calculate the attention score between the query token (x^2) and each other input token.
# A higher value in the final attention score (calculated through the dot product), means
# a greater alignment or similarity between the 2 input embedding vectors.
# tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865])
query = inputs[1] # journey (x^2)

attention_scores = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
     attention_scores[i] = torch.dot(x_i, query)
     
print("Attention scores: ", attention_scores)

# Normalize the attention scores - they will sum up to 1.0
# tensor([0.1455, 0.2278, 0.2249, 0.1285, 0.1077, 0.1656])
attention_weights1 = attention_scores / attention_scores.sum()
print("Attention weights: ", attention_weights1)
print("Sum: ", attention_weights1.sum())

# Normalize the attention scores with the softmax function
# tensor([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581]
attention_weights2 = torch.softmax(attention_scores, dim=0)
print("Attention weights (softmax): ", attention_weights2)
print("Sum: ", attention_weights2.sum())

# Calculating the context vector
# =>  tensor(0.1385) => attention weight
# =>  tensor([0.4300, 0.1500, 0.8900]) => embeddings of "Your" (x^1)
# 0.43 * 0.1385 = 0.059555
# 0.15 * 0.1385 = 0.020775
# 0.89 * 0.1385 = 0.123265

# =>  tensor(0.2379) => attention weight
# =>  tensor([0.5500, 0.8700, 0.6600]) => embeddings of "yourney" (x^2)
# 0.55 * 0.2379 = 0.130845
# 0.87 * 0.2379 = 0.206973
# 0.66 * 0.2379 = 0.157014

# =>  tensor(0.2333) => attention weight
# =>  tensor([0.5700, 0.8500, 0.6400]) => embeddings of "starts" (x^3)
# 0.57 * 0.2333 = 0.132981
# 0.85 * 0.2333 = 0.198305
# 0.64 * 0.2333 = 0.149312

# =>  tensor(0.1240) => attention weight
# =>  tensor([0.2200, 0.5800, 0.3300]) => embeddings of "with" (x^4)
# 0.22 * 0.1240 = 0.02728 
# 0.58 * 0.1240 = 0.07192
# 0.33 * 0.1240 = 0.04092

# =>  tensor(0.1082) => attention weight
# =>  tensor([0.7700, 0.2500, 0.1000]) => embeddings of "one" (x^5)
# 0.77 * 0.1082 = 0.083314
# 0.25 * 0.1082 = 0.02705
# 0.10 * 0.1082 = 0.01082

# =>  tensor(0.1581) => attention weight
# =>  tensor([0.0500, 0.8000, 0.5500]) => embeddings of "step" (x^6)
# 0.05 * 0.1581 = 0.007905 
# 0.80 * 0.1581 = 0.12648
# 0.55 * 0.1581 = 0.086955

# Final sums:
#   0.059555    0.020775    0.123265
# + 0.130845    0.206973    0.157014
# + 0.132981    0.198305    0.149312
# + 0.02728     0.07192     0.04092
# + 0.083314    0.02705     0.01082
# + 0.007905    0.12648     0.086955
# ===================================
#.  0.44188     0.651503    0.568286

# tensor([0.4419, 0.6515, 0.5683])
context_vector = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
     context_vector += attention_weights2[i] * x_i
     print("=> Attention Weight: ", attention_weights2[i])
     print("=> Token Embeddings: ", x_i)
     print("=> Context Weight:   ", attention_weights2[i] * x_i)
     print("")

print("Final Context Vector: ", context_vector)
print("")

###########################################
# Compute Attention Weights for all inputs
###########################################

# Compute the attention weights for each input
# with 2 nested loops
attention_scores = torch.empty(6, 6)

for i, x_i in enumerate(inputs):
     for j, x_j in enumerate(inputs):
        attention_scores[i, j] = torch.dot(x_i, x_j)

print("Attention scores: ", attention_scores)

# Compute the attention weights just with a simple
# matrix multiplication
attention_scores = inputs @ inputs.T
print("Attention scores: ", attention_scores)

# Normalize the attention scores
# We want to perform the normalization across the last dimension (dim=-1)
attention_weights3 = torch.softmax(attention_scores, dim=-1)
print("Attention weights: ", attention_weights3)

# Check if all rows are summing up to 1.0
print("All row sums: ", attention_weights3.sum(dim=-1))

# Calculate all context vectors
all_context_vectors = attention_weights3 @ inputs
print("All context vectors: ", all_context_vectors)
print("")
print("")

#  RUNTIME PERFORMANCE METRICS
# def attention_fn(inputs):
#     attention_scores = inputs @ inputs.T
#     attention_weights3 = torch.softmax(attention_scores, dim=-1)
#     all_context_vectors = attention_weights3 @ inputs
#     return all_context_vectors

# import torch.fx as fx

# gm = fx.symbolic_trace(attention_fn)
# gm.graph.print_tabular()

# with torch.autograd.profiler.profile(record_shapes=True) as prof:
#     attention_scores = inputs @ inputs.T
#     attention_weights3 = torch.softmax(attention_scores, dim=-1)
#     all_context_vectors = attention_weights3 @ inputs

# print(prof.key_averages().table(sort_by="cpu_time_total"))

########################################
# Self Attention with Trainable Weights
########################################
print("######################################")
print("Self Attention with Trainable Weights")
print("######################################")

# Some dummy embedding values
inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89], # Your      (x^1)
        [0.55, 0.87, 0.66], # journey   (x^2)
        [0.57, 0.85, 0.64], # starts    (x^3)
        [0.22, 0.58, 0.33], # with      (x^4)
        [0.77, 0.25, 0.10], # one       (x^5)
        [0.05, 0.80, 0.55]  # step      (x^6)
    ]
)

# Some variable definitions
x2 = inputs[1]           # journey (x^2)
d_in = inputs.shape[1]   # Input Embedding Size: 3
d_out = 2                # Output Embedding Size: 2

# Weight matrix definitions
torch.manual_seed(123)
weight_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad = False)    # 3 x 2
weight_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad = False)      # 3 x 2
weight_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad = False)    # 3 x 2

# Compute the query, key, and value vectors for x^2
query2 = x2 @ weight_query
key2 = x2 @ weight_key
value2 = x2 @ weight_value

# Compute the key and value vectors for all input elements
keys = inputs @ weight_key
values = inputs @ weight_value
print("keys.shape: ", keys.shape)
print("values.shape: ", values.shape)

# Compute one specific attention score: w22
keys2 = keys[1]
attention_score22 = query2.dot(keys2)
print("Attention score w22: ", attention_score22)

# Compute all attention scores for a given query
attention_scores2 = query2 @ keys.T
print("All Attention scores: ", attention_scores2)

# Compute the attention weights
d_k = keys.shape[-1]
attention_weights2 = torch.softmax(attention_scores2 / d_k ** 0.5, dim = -1)
print(attention_weights2)

# Compute the context vector
context_vector2 = attention_weights2 @ values
print("Context vector: ", context_vector2)

# Use the SelfAttention class
torch.manual_seed(789)
self_attention = SelfAttention(d_in, d_out)
print("Self Attention: ", self_attention(inputs))
print("")

####################################
# Self Attention - Casual Attention
####################################
print("##################################")
print("Self Attention - Casual Attention")
print("##################################")

queries = self_attention.W_query(inputs)
keys = self_attention.W_key(inputs)
attention_scores = queries @ keys.T
attention_weights = torch.softmax(attention_scores / keys.shape[-1] ** 0.5, dim = -1)
print(attention_weights)

# Create a mask where the values above the diagonal are zeros
context_length = attention_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
print(mask_simple)

# Zero out all weights above the diagonal
masked_simple = attention_weights * mask_simple
print(masked_simple)

# Renormalize the attention weights
row_sums = masked_simple.sum(dim = -1, keepdim = True)
masked_simple_normalized = masked_simple / row_sums
print(masked_simple_normalized)

# Attention weights with Dropout
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)
example = torch.ones(6, 6)
print(dropout(example))

# Apply the dropout to the attention weight matrix
torch.manual_seed(123)
print(dropout(masked_simple_normalized))
print("")

#########################
# Casual Attention Class
#########################
print("#######################")
print("Casual Attention Class")
print("#######################")

batch = torch.stack((inputs, inputs), dim = 0)
print(batch.shape)

# Use the CasualAttention class
torch.manual_seed(123)
context_length = batch.shape[1]
casual_attention = CasualAttention(d_in, d_out, context_length, 0.0)
context_vectors = casual_attention(batch)
print("context_vectors.shape: ", context_vectors.shape)
print("")

# Visualization of the model
# dot = make_dot(context_vectors, params=dict(casual_attention.named_parameters()))
# dot.render("casual_attention_graph", format="png")

#######################
# Multi-Head Attention
#######################
print("###########################")
print("Multi-Head Attention Class")
print("###########################")

torch.manual_seed(123)
context_length = batch.shape[1] # Number of tokens
d_in, d_out = 3, 2
batch = torch.stack((inputs, inputs), dim = 0)

multi_head_attention = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads = 2)
context_vectors = multi_head_attention(batch)
print(context_vectors)
print("context_vectors.shape: ", context_vectors.shape)
print("")

# Visualization of the model
# dot = make_dot(context_vectors, params=dict(multi_head_attention.named_parameters()))
# dot.render("multi_head_attention_graph", format="png")

#######################
# Multi-Head Attention
#######################
print("###########################")
print("Multi-Head Attention Class")
print("###########################")

torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
d_out = 2
multi_head_attention = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads = 2)
context_vectors = multi_head_attention(batch)
print(context_vectors)
print("context_vectors.shape: ", context_vectors.shape)
print("")