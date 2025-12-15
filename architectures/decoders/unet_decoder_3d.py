import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock3D(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock3D(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        # upsample deep feature to match skip feature resolution
        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)

        # concatenate
        x = torch.cat([x, skip], dim=1)

        # conv block
        x = self.conv(x)
        return x


class UNetDecoder3D(nn.Module):
    def __init__(self, embed_dims, num_classes):
        super().__init__()

        C1, C2, C3, C4 = embed_dims

        self.bottom = ConvBlock3D(C4, C4)

        self.up3 = UpBlock3D(C4, C3, C3)   # (f4 + f3)
        self.up2 = UpBlock3D(C3, C2, C2)   # (decode + f2)
        self.up1 = UpBlock3D(C2, C1, C1)   # (decode + f1)

        self.out_conv = nn.Conv3d(C1, num_classes, kernel_size=1)

    def forward(self, features, input_shape):
        # features: list [f1, f2, f3, f4]
        f1, f2, f3, f4 = features

        x = self.bottom(f4)
        x = self.up3(x, f3)
        x = self.up2(x, f2)
        x = self.up1(x, f1)

        logits = self.out_conv(x)
        # 입력 이미지 크기로 upsample 
        logits = F.interpolate(logits, size=input_shape, mode="trilinear", align_corners=False)

        return logits