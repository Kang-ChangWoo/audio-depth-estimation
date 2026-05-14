"""ASPP and ASFF modules for EchoDiffusion spectrogram encoding."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super().__init__(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels),
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        dy = x2.size(2) - x1.size(2)
        dx = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling."""

    def __init__(self, in_channel, depth):
        super().__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)

    def forward(self, x):
        size = x.shape[2:]
        image_features = F.interpolate(self.conv(self.mean(x)), size=size, mode='bilinear')
        net = self.conv_1x1_output(torch.cat([
            image_features,
            self.atrous_block1(x),
            self.atrous_block6(x),
            self.atrous_block12(x),
            self.atrous_block18(x),
        ], dim=1))
        return net


def add_conv(in_ch, out_ch, ksize, stride, leaky=True):
    """Conv2d + BatchNorm + activation block."""
    pad = (ksize - 1) // 2
    layers = [
        nn.Conv2d(in_ch, out_ch, ksize, stride, pad, bias=False),
        nn.BatchNorm2d(out_ch),
    ]
    if leaky:
        layers.append(nn.LeakyReLU(0.1))
    else:
        layers.append(nn.ReLU6(inplace=True))
    return nn.Sequential(*layers)


class ASFF(nn.Module):
    """Adaptively Spatial Feature Fusion."""

    def __init__(self, level, rfb=False):
        super().__init__()
        self.level = level
        self.dim = [512, 256, 128]
        self.inter_dim = self.dim[self.level]
        compress_c = 8 if rfb else 16

        if level == 0:
            self.stride_level_1 = add_conv(256, self.inter_dim, 3, 2)
            self.stride_level_2 = add_conv(128, self.inter_dim, 3, 2)
            self.expand = add_conv(self.inter_dim, 512, 3, 1)
        elif level == 1:
            self.compress_level_0 = add_conv(512, self.inter_dim, 1, 1)
            self.stride_level_2 = add_conv(128, self.inter_dim, 3, 2)
            self.expand = add_conv(self.inter_dim, 256, 3, 1)
        elif level == 2:
            self.compress_level_0 = add_conv(512, self.inter_dim, 1, 1)
            self.compress_level_1 = add_conv(256, self.inter_dim, 1, 1)
            self.expand = add_conv(self.inter_dim, 128, 3, 1)

        self.weight_level_0 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_levels = nn.Conv2d(compress_c * 3, 3, 1, 1, 0)

    def forward(self, x_level_0, x_level_1, x_level_2):
        if self.level == 0:
            l0 = x_level_0
            l1 = self.stride_level_1(x_level_1)
            l2 = self.stride_level_2(F.max_pool2d(x_level_2, 3, stride=2, padding=1))
        elif self.level == 1:
            l0 = F.interpolate(self.compress_level_0(x_level_0), scale_factor=2, mode='nearest')
            l1 = x_level_1
            l2 = self.stride_level_2(x_level_2)
        elif self.level == 2:
            l0 = F.interpolate(self.compress_level_0(x_level_0), scale_factor=4, mode='nearest')
            l1 = F.interpolate(self.compress_level_1(x_level_1), scale_factor=2, mode='nearest')
            l2 = x_level_2

        w = F.softmax(self.weight_levels(torch.cat([
            self.weight_level_0(l0), self.weight_level_1(l1), self.weight_level_2(l2),
        ], dim=1)), dim=1)

        fused = l0 * w[:, 0:1] + l1 * w[:, 1:2] + l2 * w[:, 2:]
        return self.expand(fused)


class UNetASPPASFF(nn.Module):
    """Small UNet with ASPP and ASFF for spectrogram encoding.
    Input: (B, 2, 128, 128) spectrogram -> Output: (B, 128, 32, 32) features."""

    def __init__(self, in_channels=2, base_c=64, bilinear=False):
        super().__init__()
        self.in_conv = Down(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)

        self.aspp1 = ASPP(64, 64)
        self.aspp2 = ASPP(128, 128)
        self.aspp3 = ASPP(256, 256)
        self.aspp4 = ASPP(512, 512)

        self.asff_0 = ASFF(level=0)
        self.asff_1 = ASFF(level=1)
        self.asff_2 = ASFF(level=2)

        self.up2 = Up(base_c * 8, base_c * 4, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2, bilinear)

    def forward(self, x):
        x1 = self.aspp1(self.in_conv(x))        # (B, 64, 64, 64)
        x2 = self.aspp2(self.down1(x1))          # (B, 128, 32, 32)
        x3 = self.aspp3(self.down2(x2))          # (B, 256, 16, 16)
        x4 = self.aspp4(self.down3(x3))          # (B, 512, 8, 8)

        f0 = self.asff_0(x4, x3, x2)            # (B, 512, 8, 8)
        f1 = self.asff_1(x4, x3, x2)            # (B, 256, 16, 16)
        f2 = self.asff_2(x4, x3, x2)            # (B, 128, 32, 32)

        x = self.up2(f0, f1)                     # (B, 256, 16, 16) -> (B, 128, 32, 32)
        x = self.up3(x, f2)                      # (B, 128, 32, 32)
        return x
