import torch

x = torch.rand(3, 3).to("cuda")
print(x.device)