import torch

A = torch.tensor([[65504, 1, -65504, 1]], dtype=torch.float16, device="cuda")
B = torch.tensor([[1], [1e-3], [1], [1e-3]], dtype=torch.float16, device="cuda")

print(A @ B)
