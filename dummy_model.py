import torch
import torch.nn as nn

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        batch = x.shape[0]
        # Output shape must match your encode: (8,8,73)
        policy = torch.ones((batch, 8, 8, 73)) / (8*8*73)
        value = torch.zeros((batch, 1))
        return policy, value