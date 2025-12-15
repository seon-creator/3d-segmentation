import torch
from torch import nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channel, embed_dim, kernel_size, stride, padding):
        super().__init__()
        self.patch_embeddings = nn.Conv3d(
            in_channel,                 # in-channel
            embed_dim,                  # out-channel
            kernel_size=kernel_size,    # NxN kernel-size
            stride=stride,              # stride
            padding=padding             # zero-padding (padding_mode='zeros'; default)
        )
        # nn.LayerNorm(normalized_shape, 
        #              eps=1e-05,                   => 분모=0 방지
        #              elementwise_affine=True,     => Learnable gamma, bias 매개변수를 가짐
        #              bias=True,                   => bias 학습 여부 결정, True: bias 학습, False: gamma만 사용
        #              device=None, 
        #              dtype=None)
        self.norm = nn.LayerNorm(embed_dim)  # initiate class

    def forward(self, x):
        patches = self.patch_embeddings(x)
        # torch.flatten(input, start_dim=0, end_dim=-1)
        # patches.shape = (Batch size, embedding dim, Depth, Height, Width)
        # ex) patches.shape = (2, 32, 8, 8, 8)
        # ex) patches.flatten(start_dim=2): (2, 32, 512)
        # ex) patches.flatten(start_dim=2).transpose(1, 2): (2, 512, 32)
        patches = patches.flatten(start_dim=2).transpose(1, 2)
        # transformer input requestion
        # (Batch size, Sequence Length, Embedding Dimension)
        patches = self.norm(patches)
        return patches