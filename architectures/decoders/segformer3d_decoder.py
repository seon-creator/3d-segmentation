import torch
from torch import nn
from architectures.decoders.mlp import MLP_

class SegFormerDecoderHead(nn.Module):
    def __init__(
        self,
        input_feature_dims,
        decoder_head_embedding_dim,
        num_classes,
        dropout=0.0,
    ):
        super().__init__()

        C4, C3, C2, C1 = input_feature_dims

        self.linear_c4 = MLP_(input_dim=C4, embed_dim=decoder_head_embedding_dim)
        self.linear_c3 = MLP_(input_dim=C3, embed_dim=decoder_head_embedding_dim)
        self.linear_c2 = MLP_(input_dim=C2, embed_dim=decoder_head_embedding_dim)
        self.linear_c1 = MLP_(input_dim=C1, embed_dim=decoder_head_embedding_dim)

        self.linear_fuse = nn.Sequential(
            nn.Conv3d(in_channels=4 * decoder_head_embedding_dim, 
                      out_channels=decoder_head_embedding_dim, 
                      kernel_size=1, 
                      bias=False),
            nn.BatchNorm3d(decoder_head_embedding_dim),
            nn.ReLU(),
        )

        self.dropout = nn.Dropout(dropout)

        self.linear_pred = nn.Conv3d(decoder_head_embedding_dim, num_classes, kernel_size=1)

        self.upsample_volume = nn.Upsample(scale_factor=4.0, mode="trilinear", align_corners=False)

    def forward(self, c1, c2, c3, c4):
        B = c4.shape[0]

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(B, -1, *c4.shape[2:]).contiguous()
        _c4 = nn.functional.interpolate(_c4, size=c1.shape[2:], mode="trilinear", align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(B, -1, *c3.shape[2:]).contiguous()
        _c3 = nn.functional.interpolate(_c3, size=c1.shape[2:], mode="trilinear", align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(B, -1, *c2.shape[2:]).contiguous()
        _c2 = nn.functional.interpolate(_c2, size=c1.shape[2:], mode="trilinear", align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(B, -1, *c1.shape[2:]).contiguous()

        fused = self.linear_fuse(torch.cat([_c4,_c3,_c2,_c1], dim=1))
        fused = self.dropout(fused)
        out = self.linear_pred(fused)
        out = self.upsample_volume(out)
        return out


# ======================================================================
# SegFormer Decoder + 🎯 ET Auxiliary Head
# ======================================================================
class SegFormerDecoderHead_ETAux(nn.Module):
    """
    SegFormer decoder + ET-only auxiliary head
    """

    def __init__(
        self,
        input_feature_dims,
        decoder_head_embedding_dim,
        num_classes,
        dropout=0.0,
    ):
        super().__init__()

        C4, C3, C2, C1 = input_feature_dims

        # shared projection layers
        self.linear_c4 = MLP_(input_dim=C4, embed_dim=decoder_head_embedding_dim)
        self.linear_c3 = MLP_(input_dim=C3, embed_dim=decoder_head_embedding_dim)
        self.linear_c2 = MLP_(input_dim=C2, embed_dim=decoder_head_embedding_dim)
        self.linear_c1 = MLP_(input_dim=C1, embed_dim=decoder_head_embedding_dim)

        # fusion
        self.linear_fuse = nn.Sequential(
            nn.Conv3d(
                in_channels=4 * decoder_head_embedding_dim,
                out_channels=decoder_head_embedding_dim,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm3d(decoder_head_embedding_dim),
            nn.ReLU(),
        )

        self.dropout = nn.Dropout(dropout)

        # main segmentation (WT, TC, ET)
        self.linear_pred = nn.Conv3d(decoder_head_embedding_dim, num_classes, 1)

        # 🎯 Auxiliary ET head
        self.linear_pred_et = nn.Conv3d(decoder_head_embedding_dim, 1, 1)

        # upsample both
        self.upsample_volume = nn.Upsample(
            scale_factor=4.0, mode="trilinear", align_corners=False
        )

    def forward(self, c1, c2, c3, c4):
        B = c4.shape[0]

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(B, -1, *c4.shape[2:])
        _c4 = nn.functional.interpolate(_c4, size=c1.shape[2:], mode="trilinear", align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(B, -1, *c3.shape[2:])
        _c3 = nn.functional.interpolate(_c3, size=c1.shape[2:], mode="trilinear", align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(B, -1, *c2.shape[2:])
        _c2 = nn.functional.interpolate(_c2, size=c1.shape[2:], mode="trilinear", align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(B, -1, *c1.shape[2:])

        fused = self.linear_fuse(torch.cat([_c4,_c3,_c2,_c1], dim=1))
        fused = self.dropout(fused)

        # main segmentation
        seg_out = self.linear_pred(fused)
        seg_out = self.upsample_volume(seg_out)

        # ET-only auxiliary output
        et_out = self.linear_pred_et(fused)
        et_out = self.upsample_volume(et_out)

        return seg_out, et_out