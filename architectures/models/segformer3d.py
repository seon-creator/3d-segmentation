import math
from typing import Dict, Optional, List

import torch
from torch import nn

# -----------------------------
# encoder / decoder import 경로
# -----------------------------
from architectures.encoders.segformer3d_encoder import MixVisionTransformer
from architectures.decoders.segformer3d_decoder import SegFormerDecoderHead


def build_segformer3d_model(config: Optional[Dict] = None) -> nn.Module:
    """
    Factory function used in build_architecture.py
    """
    params = config["model_parameters"]

    return SegFormer3D(
        in_channels=params["in_channels"],      # modalities 수
        sr_ratios=params["sr_ratios"],          # 
        embed_dims=params["embed_dims"],
        patch_kernel_size=params["patch_kernel_size"],
        patch_stride=params["patch_stride"],
        patch_padding=params["patch_padding"],
        mlp_ratios=params["mlp_ratios"],
        num_heads=params["num_heads"],
        depths=params["depths"],
        decoder_head_embedding_dim=params["decoder_head_embedding_dim"],
        num_classes=params["num_classes"],
        decoder_dropout=params["decoder_dropout"],
    )


class SegFormer3D(nn.Module):
    """
    최종 SegFormer3D 모델 클래스
    Encoder (MixVisionTransformer) + Decoder (SegFormerDecoderHead)
    """

    def __init__(
        self,
        in_channels: int = 4,
        sr_ratios: List[int] = [4, 2, 1, 1],
        embed_dims: List[int] = [32, 64, 160, 256],
        patch_kernel_size: List[int] = [7, 3, 3, 3],
        patch_stride: List[int] = [4, 2, 2, 2],
        patch_padding: List[int] = [3, 1, 1, 1],
        mlp_ratios: List[int] = [4, 4, 4, 4],
        num_heads: List[int] = [1, 2, 5, 8],
        depths: List[int] = [2, 2, 2, 2],
        decoder_head_embedding_dim: int = 256,
        num_classes: int = 3,
        decoder_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # ---------------- Encoder ----------------
        self.encoder = MixVisionTransformer(
            in_channels=in_channels,
            sr_ratios=sr_ratios,
            embed_dims=embed_dims,
            patch_kernel_size=patch_kernel_size,
            patch_stride=patch_stride,
            patch_padding=patch_padding,
            mlp_ratios=mlp_ratios,
            num_heads=num_heads,
            depths=depths,
        )

        # ---------------- Decoder ----------------
        reversed_embed_dims = embed_dims[::-1]
        self.decoder = SegFormerDecoderHead(
            input_feature_dims=reversed_embed_dims,
            decoder_head_embedding_dim=decoder_head_embedding_dim,
            num_classes=num_classes,
            dropout=decoder_dropout,
        )

        # weight init
        self.apply(self._init_weights)

    # ----------------------------------------------------------------------
    # weight initialization (원본 코드 유지)
    # ----------------------------------------------------------------------
    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

        elif isinstance(m, nn.Conv3d):
            fan_out = (
                m.kernel_size[0]
                * m.kernel_size[1]
                * m.kernel_size[2]
                * m.out_channels
            )
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    # ----------------------------------------------------------------------
    # forward
    # ----------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, D, H, W)

        Returns:
            segmentation map: (B, num_classes, D, H, W)
        """
        c1, c2, c3, c4 = self.encoder(x)
        out = self.decoder(c1, c2, c3, c4)
        return out