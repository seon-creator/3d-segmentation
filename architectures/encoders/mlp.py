import torch
from torch import nn
from architectures.encoders.attention import cube_root

class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, groups=dim)
        self.bn = nn.BatchNorm3d(dim)

    def forward(self, x):
        B, N, C = x.shape
        n = cube_root(N)
        x = x.transpose(1,2).reshape(B,C,n,n,n)
        x = self.dwconv(x)
        x = self.bn(x)
        x = x.flatten(2).transpose(1,2)
        return x


class MLP(nn.Module):
    def __init__(self, in_feature, mlp_ratio, dropout=0.0):
        super().__init__()
        hidden = in_feature * mlp_ratio
        self.fc1 = nn.Linear(in_feature, hidden)
        self.dwconv = DWConv(hidden)
        self.fc2 = nn.Linear(hidden, in_feature)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x