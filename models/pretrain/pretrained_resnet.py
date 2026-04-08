"""Pretrained ResNet-50 adapted for audio-to-depth estimation.

Uses a learnable 1×1 conv to project 2-channel binaural spectrogram
to 3-channel pseudo-RGB, then feeds through a pretrained ResNet-50
encoder.  Multi-scale features from layers 1-4 are progressively
upsampled via a lightweight decoder with skip connections to produce
a dense depth map at the original resolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from torchvision.models import ResNet50_Weights


class DepthDecoder(nn.Module):
    """Progressive upsampling decoder with skip connections from encoder."""

    def __init__(self, depth_norm=True):
        super().__init__()
        self.depth_norm = depth_norm

        # Reduce layer4 channels
        self.reduce4 = nn.Sequential(
            nn.Conv2d(2048, 512, 1), nn.BatchNorm2d(512), nn.ReLU(True))

        # Up stages: each takes concat(upsampled, skip) → out
        # stage3: 512+1024 → 256
        self.up3 = self._up_block(512 + 1024, 256)
        # stage2: 256+512 → 128
        self.up2 = self._up_block(256 + 512, 128)
        # stage1: 128+256 → 64
        self.up1 = self._up_block(128 + 256, 64)
        # stage0: 64+64 → 32
        self.up0 = self._up_block(64 + 64, 32)

        self.head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 1, 1),
        )

    @staticmethod
    def _up_block(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True),
        )

    def forward(self, feats, target_h, target_w):
        """
        feats : dict with keys 'stem', 'layer1', 'layer2', 'layer3', 'layer4'
        """
        x = self.reduce4(feats['layer4'])          # (B, 512, H/32, W/32)

        x = F.interpolate(x, size=feats['layer3'].shape[2:], mode='bilinear',
                          align_corners=False)
        x = self.up3(torch.cat([x, feats['layer3']], dim=1))

        x = F.interpolate(x, size=feats['layer2'].shape[2:], mode='bilinear',
                          align_corners=False)
        x = self.up2(torch.cat([x, feats['layer2']], dim=1))

        x = F.interpolate(x, size=feats['layer1'].shape[2:], mode='bilinear',
                          align_corners=False)
        x = self.up1(torch.cat([x, feats['layer1']], dim=1))

        x = F.interpolate(x, size=feats['stem'].shape[2:], mode='bilinear',
                          align_corners=False)
        x = self.up0(torch.cat([x, feats['stem']], dim=1))

        x = self.head(x)
        if self.depth_norm:
            x = torch.sigmoid(x)

        if x.shape[2] != target_h or x.shape[3] != target_w:
            x = F.interpolate(x, (target_h, target_w), mode='bilinear',
                              align_corners=False)
        return x


class PretrainedResNet(nn.Module):
    """Pretrained ResNet-50 encoder + multi-scale decoder for audio-to-depth.

    Pipeline:
      1. Input adapter — Conv2d(2, 3, 1×1): spectrogram → pseudo-RGB
      2. ResNet-50 encoder (ImageNet pretrained), extract multi-scale features
      3. FPN-style decoder with skip connections → depth map

    Encoder feature scales (for 256×512 input):
      stem   (64,  128, 256)  — after conv1+bn+relu+maxpool
      layer1 (256,  64, 128)  — /4
      layer2 (512,  32,  64)  — /8
      layer3 (1024, 16,  32)  — /16
      layer4 (2048,  8,  16)  — /32

    Parameters
    ----------
    cfg : SimpleNamespace
        Needs ``cfg.dataset.depth_norm``.
    input_nc : int
        Audio channels (2).
    pretrained : bool
        Load ImageNet weights.
    freeze_encoder : bool
        Freeze ResNet encoder weights.
    """

    def __init__(self, cfg, input_nc=2, pretrained=True, freeze_encoder=False):
        super().__init__()
        depth_norm = getattr(cfg.dataset, 'depth_norm', True)

        # (1) Input adapter: 2ch → 3ch
        self.input_adapter = nn.Conv2d(input_nc, 3, kernel_size=1)

        # (2) Pretrained ResNet-50 encoder
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        resnet = tv_models.resnet50(weights=weights)

        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        if freeze_encoder:
            for module in [self.stem, self.layer1, self.layer2,
                           self.layer3, self.layer4]:
                for p in module.parameters():
                    p.requires_grad = False

        # (3) Decoder
        self.decoder = DepthDecoder(depth_norm=depth_norm)

    def forward(self, x):
        """
        x : (B, 2, H, W) binaural spectrogram
        Returns: (B, 1, H, W) depth
        """
        orig_h, orig_w = x.shape[2], x.shape[3]

        x = self.input_adapter(x)                  # (B, 3, H, W)

        # Encoder with multi-scale features
        s = self.stem(x)                            # (B, 64, H/4, W/4)
        l1 = self.layer1(s)                         # (B, 256, H/4, W/4)
        l2 = self.layer2(l1)                        # (B, 512, H/8, W/8)
        l3 = self.layer3(l2)                        # (B, 1024, H/16, W/16)
        l4 = self.layer4(l3)                        # (B, 2048, H/32, W/32)

        feats = {'stem': s, 'layer1': l1, 'layer2': l2,
                 'layer3': l3, 'layer4': l4}

        depth = self.decoder(feats, orig_h, orig_w)
        return depth
