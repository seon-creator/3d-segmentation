import torch
from torch import nn

class MLP_(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1,2)
        x = self.proj(x)
        x = self.norm(x)
        return x