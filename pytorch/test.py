import torch

print("PyTorch version:")
print(torch.__version__)
print("") 

print("CUDA available?")
print(torch.cuda.is_available())
print("")

print("Apple GPU available?")
print(torch.backends.mps.is_available())

# Create a 0D tensor
tensor0d = torch.tensor(1.0)
print("Tensor 0D:")
print(tensor0d)
print(tensor0d.dtype)
print("")

# Create a 1D tensor
tensor1d = torch.tensor([1, 2, 3, 4])
print("Tensor 1D:")
print(tensor1d)
print(tensor1d.dtype)
print("")
