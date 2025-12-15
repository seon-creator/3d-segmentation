import math
import torch
from torch import nn

def cube_root(n):
    return round(n ** (1/3))


class SelfAttention(nn.Module):
    def __init__(
        self, 
        embed_dim,          # embedding-dimension
        num_heads,          # The number of heads from multi-head attention
        sr_ratio,           # Spatial reduction ratio ex) sr_ratios: [4, 2, 1, 1]
        qkv_bias=False,     # The option for adding bias of Query, Key, Value
        attn_dropout=0.0,   # attention dropout ratio
        proj_dropout=0.0    # final projection ratio
    ):
        super().__init__()
        assert embed_dim % num_heads == 0   # embedding dimention should be divided up by num_heads

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # head 차원 = embedding 차원 / head 수

        # nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
        self.query = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.key_value = nn.Linear(embed_dim, embed_dim * 2, bias=qkv_bias)

        # Dropout
        # nn.Droptou(p=0.5, inplace=False)
        # inplace=True : Dropout 연산을 입력 텐서가 지정된 기존 메모리 공간(in-place) 에서 수행
        # inplace=False : 입력 텐서를 복사하여 새 텐서를 생성하고 Dropout 연산을 새 텐서 위에서 수행
        self.attn_drop = nn.Dropout(attn_dropout)   # class initiation

        # Linear projection
        self.proj = nn.Linear(embed_dim, embed_dim) # class initiation
        self.proj_drop = nn.Dropout(proj_dropout)   # class initiation

        self.sr_ratio = sr_ratio

        # Spatial reduction ratio
        if sr_ratio > 1:
            # in-out dimension not changed, feature map size -> divided by sr_ratio
            self.sr = nn.Conv3d(embed_dim, embed_dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # B: Batch size, N: sequence length, C: Embedding dimension 
        B, N, C = x.shape

        # linear layer create, reshaping to appropriate shape of multihead attention
        # self.query(x).reshape: 입력 텐서 -> (B, N, Num_heads, Single-head_dim)
        # .permute(0,2,1,3): shape 순서 변환 (B, N, Num_heads, Single-head_dim) -> (B, Num_heads, N, Single-head_dim)
        # Attention 연산은 Batch size, Num_head가 앞쪽에 위치할 때 효율적임 (why?)
        q = self.query(x).reshape(B, N, self.num_heads, self.head_dim).permute(0,2,1,3)

        if self.sr_ratio > 1:
            n = cube_root(N)
            # x.permute(0, 2, 1).reshape(B, C, n,n,n): (B, N, C) -> (B, C, N) -> (B, C, n,n,n)
            x_ = x.permute(0,2,1).reshape(B, C, n,n,n)
            # (B, C, n,n,n) -> (B, C, n/sr,n/sr,n/sr) -> (B, C, N) -> (B, N, C)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0,2,1)
            # LayerNormalization
            x_ = self.norm(x_)

            # self.key_value = nn.Linear(embed_dim, embed_dim * 2, bias=qkv_bias)
            # (B, N, 2, num_heads, Single-head_dim) -> (2, B, Num_heads, N, Single-head_dim)        
            kv = (
                self.key_value(x_)
                .reshape(B, -1, 2, self.num_heads, self.head_dim)
                .permute(2,0,3,1,4)
            )
        else:
            kv = (
                self.key_value(x)
                .reshape(B, -1, 2, self.num_heads, self.head_dim)
                .permute(2,0,3,1,4)
            )
        k, v = kv[0], kv[1]

        # q.shape : (B, Num_heads, N, Single-head_dim)
        # k.shape : (B, Num_heads, N, Single-head_dim)
        # k.transpose(-2,-1): 끝에서 두 번째, 맨 끝 transpose -> (B, Num_heads, Single-head_dim, N)
        # q @ k : matrix multiplication -> (B, Num_heads, N, N)
        # softmax 함수 통과 전 값의 분산 안정화를 위한 head-dimention의 제곱근 계산
        attn = (q @ k.transpose(-2,-1)) / math.sqrt(self.head_dim)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn) # Dropout 수행 (default=0.0)

        # attn.shape: (B, Num_heads, N, N)
        # v.shape: (B, Num_heads, N, Single-head_dim)
        # attn @ v -> (B, Num_heads, N, Single-head_dim)
        # (attn @ v).transpose(1,2) -> (B, N, Num_heads, Single-head_dim)
        # (attn @ v).transpose(1,2).reshape(B,N,C) -> (B, N, C) C : Num_heads x Single-head_dim
        out = (attn @ v).transpose(1,2).reshape(B,N,C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out