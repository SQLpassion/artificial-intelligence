from torch.utils.data import Dataset, DataLoader
import tiktoken
import torch

class GPTDataset(Dataset):
    # Constructor
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        
        # Tokenize the provided input text
        token_ids = tokenizer.encode(text)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    # Returns the total number of rows in the Dataset
    def __len__(self):
        return len(self.input_ids)
    
    # Returns a single row from the Dataset
    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]