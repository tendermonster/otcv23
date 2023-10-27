import torch
from torch.nn import functional as F

a = torch.tensor([[1, 2, 3],[1, 2, 3],[1, 2, 3]], dtype=torch.float32)
print(a.shape)

N = F.softmax(a, dim=-1)
# N = F.softmax(a)
print(N)