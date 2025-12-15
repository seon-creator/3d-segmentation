import torch
from torch import nn

from architectures.encoders.patch_embedding import PatchEmbedding
from architectures.encoders.transformer_blocks import TransformerBlock
from architectures.encoders.attention import cube_root


class MixVisionTransformer(nn.Module):
    def __init__(
        self,
        in_channels,        # 모달리티 수 (2: T1ce, Flair)
        sr_ratios,          # 
        embed_dims,         # feature map을 임베딩할 크기
        patch_kernel_size,  # NxNxN 크기의 patch
        patch_stride,       # 패치의 stride
        patch_padding,      # zero-padding 
        mlp_ratios,         # 
        num_heads,          # The number of heads
        depths,             # depth
    ):
        super().__init__()

        # embedding block
        self.embed_1 = PatchEmbedding(in_channels, embed_dims[0], patch_kernel_size[0], patch_stride[0], patch_padding[0])
        self.embed_2 = PatchEmbedding(embed_dims[0], embed_dims[1], patch_kernel_size[1], patch_stride[1], patch_padding[1])
        self.embed_3 = PatchEmbedding(embed_dims[1], embed_dims[2], patch_kernel_size[2], patch_stride[2], patch_padding[2])
        self.embed_4 = PatchEmbedding(embed_dims[2], embed_dims[3], patch_kernel_size[3], patch_stride[3], patch_padding[3])


        # Transformer block
        # torch.nn.ModuleList(modules=None)
        self.tf_block1 = nn.ModuleList([TransformerBlock(embed_dims[0], num_heads[0], mlp_ratios[0], sr_ratios[0]) for _ in range(depths[0])])
        self.tf_block2 = nn.ModuleList([TransformerBlock(embed_dims[1], num_heads[1], mlp_ratios[1], sr_ratios[1]) for _ in range(depths[1])])
        self.tf_block3 = nn.ModuleList([TransformerBlock(embed_dims[2], num_heads[2], mlp_ratios[2], sr_ratios[2]) for _ in range(depths[2])])
        self.tf_block4 = nn.ModuleList([TransformerBlock(embed_dims[3], num_heads[3], mlp_ratios[3], sr_ratios[3]) for _ in range(depths[3])])

        # Layer Normalization: mean=0, std=1
        self.norm1 = nn.LayerNorm(embed_dims[0])
        self.norm2 = nn.LayerNorm(embed_dims[1])
        self.norm3 = nn.LayerNorm(embed_dims[2])
        self.norm4 = nn.LayerNorm(embed_dims[3])

    def forward(self, x):
        outputs = []

        # Stage 1
        x = self.embed_1(x)
        B, N, C = x.shape
        n = cube_root(N)
        for blk in self.tf_block1: x = blk(x)
        x = self.norm1(x).reshape(B,n,n,n,C).permute(0,4,1,2,3).contiguous()
        outputs.append(x)

        # Stage 2
        x = self.embed_2(x)
        B, N, C = x.shape
        n = cube_root(N)
        for blk in self.tf_block2: x = blk(x)
        x = self.norm2(x).reshape(B,n,n,n,C).permute(0,4,1,2,3)
        outputs.append(x)

        # Stage 3
        x = self.embed_3(x)
        B, N, C = x.shape
        n = cube_root(N)
        for blk in self.tf_block3: x = blk(x)
        x = self.norm3(x).reshape(B,n,n,n,C).permute(0,4,1,2,3)
        outputs.append(x)

        # Stage 4
        x = self.embed_4(x)
        B, N, C = x.shape
        n = cube_root(N)
        for blk in self.tf_block4: x = blk(x)
        x = self.norm4(x).reshape(B,n,n,n,C).permute(0,4,1,2,3)
        outputs.append(x)

        return outputs