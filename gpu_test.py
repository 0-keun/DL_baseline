import torch
a = torch.randn(5000, 5000, device="cuda")
b = torch.matmul(a, a)
print("Done on:", b.device)