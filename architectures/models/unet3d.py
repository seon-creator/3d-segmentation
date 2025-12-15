import torch
import torch.nn as nn
import math


##################################################################################################
# Build function (SegFormer3D 스타일과 동일한 구조)
##################################################################################################
def build_unet3d_model(config=None):
    model = UNet3D(
        in_channels=config["model_parameters"]["in_channels"],
        num_classes=config["model_parameters"]["num_classes"],
        base_channels=config["model_parameters"]["base_channels"],
        use_batchnorm=config["model_parameters"].get("use_batchnorm", True),
    )
    return model


##################################################################################################
# U-Net 3D Architecture
##################################################################################################
class UNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 3,
        base_channels: int = 32,
        use_batchnorm: bool = True,
    ):
        """
        3D U-Net Architecture
        ---------------------
        Args:
            in_channels (int): Input modality count (e.g., Flair+T1ce → 2)
            num_classes (int): Number of segmentation classes
            base_channels (int): Number of channels in first encoder layer
            use_batchnorm (bool): Use BatchNorm3D after conv layers
        """
        super().__init__()
        self.use_bn = use_batchnorm

        # ---------------- Encoder ----------------
        self.enc1 = DoubleConv3D(in_channels, base_channels, use_bn=use_batchnorm)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = DoubleConv3D(base_channels, base_channels * 2, use_bn=use_batchnorm)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = DoubleConv3D(base_channels * 2, base_channels * 4, use_bn=use_batchnorm)
        self.pool3 = nn.MaxPool3d(2)

        self.enc4 = DoubleConv3D(base_channels * 4, base_channels * 8, use_bn=use_batchnorm)
        self.pool4 = nn.MaxPool3d(2)

        # ---------------- Bottleneck ----------------
        self.bottleneck = DoubleConv3D(base_channels * 8, base_channels * 16, use_bn=use_batchnorm)

        # ---------------- Decoder ----------------
        self.upconv4 = nn.ConvTranspose3d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv3D(base_channels * 16, base_channels * 8, use_bn=use_batchnorm)

        self.upconv3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv3D(base_channels * 8, base_channels * 4, use_bn=use_batchnorm)

        self.upconv2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv3D(base_channels * 4, base_channels * 2, use_bn=use_batchnorm)

        self.upconv1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = DoubleConv3D(base_channels * 2, base_channels, use_bn=use_batchnorm)

        # ---------------- Final Output Layer ----------------
        self.out_conv = nn.Conv3d(base_channels, num_classes, kernel_size=1)

        self.apply(self._init_weights)

    ##################################################################################################
    # weight initialization (SegFormer 스타일)
    ##################################################################################################
    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    ##################################################################################################
    # forward pass
    ##################################################################################################
    def forward(self, x):
        # ---------------- Encoder ----------------
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # ---------------- Bottleneck ----------------
        b = self.bottleneck(self.pool4(e4))

        # ---------------- Decoder ----------------
        d4 = self.upconv4(b)
        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)

        out = self.out_conv(d1)
        return out


##################################################################################################
# DoubleConv3D Block (Conv3D + BN + ReLU x2)
##################################################################################################
class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=True):
        super().__init__()
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        ]
        if use_bn:
            layers.append(nn.BatchNorm3d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False))
        if use_bn:
            layers.append(nn.BatchNorm3d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


##################################################################################################
# Test Code
##################################################################################################
if __name__ == "__main__":
    input_tensor = torch.randn(1, 4, 128, 128, 128).cuda()  # (B, C, D, H, W)
    model = UNet3D(in_channels=4, num_classes=3, base_channels=32).cuda()
    output = model(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")