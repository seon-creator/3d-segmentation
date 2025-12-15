import math
import torch
import torch.nn as nn

# Encoder: SegFormer3D Encoder (MiT)
from architectures.encoders.segformer3d_encoder import MixVisionTransformer
# Decoder: 3D U-Net style decoder
from architectures.decoders.unet_decoder_3d import UNetDecoder3D

# ============================================================
# 📌 build function used by build_architecture(...)
# ============================================================

def build_segformer3d_unet_model(config=None) -> nn.Module:
    """
    Factory function to build SegFormer3D_UNetModel from config dict.
    Expects `config["model_parameters"]` to contain all necessary keys.
    """
    model_params = config["model_parameters"]
    model = SegFormer3D_UNetModel(model_params)
    return model

class SegFormer3D_UNetModel(nn.Module):
    """
    SegFormer3D Encoder (MixVisionTransformer) + UNet-style Decoder
    - Encoder: outputs [f1, f2, f3, f4]
    - Decoder: UNetDecoder3D takes these features and reconstructs full-res logits
    """

    def __init__(self, model_params: dict):
        super().__init__()

        # --------------------------
        # Encoder (SegFormer / MiT)
        # --------------------------
        self.encoder = MixVisionTransformer(
            in_channels=model_params["in_channels"],
            sr_ratios=model_params["sr_ratios"],
            embed_dims=model_params["embed_dims"],
            patch_kernel_size=model_params["patch_kernel_size"],
            patch_stride=model_params["patch_stride"],
            patch_padding=model_params["patch_padding"],
            mlp_ratios=model_params["mlp_ratios"],
            num_heads=model_params["num_heads"],
            depths=model_params["depths"],
        )

        # --------------------------
        # Decoder (UNet-style)
        # --------------------------
        self.decoder = UNetDecoder3D(
            embed_dims=model_params["embed_dims"],
            num_classes=model_params["num_classes"],
        )

        # 동일한 초기화 전략 사용 (기존 SegFormer3D와 동일한 스타일)
        self.apply(self._init_weights)

    # -----------------------------------------------------
    # Weight initialization (SegFormer3D와 동일한 로직 유지)
    # -----------------------------------------------------
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
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

    # -----------------------------------------------------
    # Forward
    # -----------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, D, H, W)
        returns: logits (B, num_classes, D, H, W)
        """
        input_shape = x.shape[2:]          # (D, H, W)
        features = self.encoder(x)         # [f1, f2, f3, f4]
        logits = self.decoder(features, input_shape=input_shape)
        return logits