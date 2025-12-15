import torch
from torch import nn

from architectures.encoders.attention import SelfAttention
from architectures.encoders.mlp import MLP

class TransformerBlock(nn.Module):
    def __init__(self, 
                embed_dim,  # input-dimension
                num_heads,  # The number of heads of Multi-head attention
                mlp_ratio,  # Multi-layer-perceptron size ex) config.yaml: mlp_ratio = [4, 4, 4, 4]
                sr_ratio):  # Spatial Reduction ratio
        super().__init__()
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(embed_dim)

        # SelfAttention
        self.attn = SelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            sr_ratio=sr_ratio,
            qkv_bias=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x