from torch.utils.data import DataLoader
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