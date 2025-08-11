from SimpleTokenizer import SimpleTokenizer
from pathlib import Path
import re

# Read a sample text from the file system
script_dir = Path(__file__).resolve().parent
file_path = script_dir / "The_Verdict.txt"

with file_path.open("r", encoding="utf-8") as file:
     raw_text = file.read()
print("Total number of characters: ", len(raw_text))
print(raw_text[:100])  # Display the first 100 characters to check the content

# Tokenize the text through Regular Expressions
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print("Total number of tokens: ", len(preprocessed))
print(preprocessed[:30])

# Create the vocabulary and assign to each unique word a token ID
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unknown|>"])
vocabulary_size = len(all_tokens)
vocabulary = {token:integer for integer, token in enumerate(all_tokens)}
print("Vocabulary size: ", vocabulary_size)

# Print out the first 50 entries with their token IDs
for i, item in enumerate(vocabulary.items()):
     print(item)

     if i >= 50:
          break

# Print out the last 5 tokens.
# It will include the special tokens "<|endoftext|>" and "<|unknown|>"
for i, item in enumerate(list(vocabulary.items())[-5:]):
     print(item)
     
# Use the SimpleTokenizer class
tokenizer = SimpleTokenizer(vocabulary)
text = """"It's the last he painted, you know,"
    Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))
print("")

# Working with the special tokens "<|endoftext|>"
# and "<|unknown|>"
text1 = "Hello, do you like tea?"               # "Hello" is not part of the vocabulary
text2 = "In the sunlit terraces of the palace." # "palace" is not part of the vocabulary
text = " <|endoftext|> ".join((text1, text2))
print(text)
print(tokenizer.encode(text))
print(tokenizer.decode(tokenizer.encode(text)))