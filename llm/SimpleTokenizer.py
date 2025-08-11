import re

class SimpleTokenizer:
    # Constructor
    def __init__(self, vocabulary):
        self.str_to_int = vocabulary
        self.int_to_str = {i:s for s, i in vocabulary.items()}

    # Encodes a given text to token IDs
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]

        # Add the special token "<|unknown|>" for unknown words
        preprocessed = [item if item in self.str_to_int
                        else "<|unknown|>" for item in preprocessed]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    # Decodes given token IDs back to text
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text