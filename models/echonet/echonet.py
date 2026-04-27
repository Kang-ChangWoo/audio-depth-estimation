"""Echo-Net: Full architecture from 'Beyond Image to Depth' (Parida et al., CVPR 2021).

All 5 subnetworks are present:
  1. Echo SubNet    – audio encoder-decoder (echo → depth)
  2. Visual SubNet  – UNet encoder-decoder, image → depth
  3. Material SubNet– ResNet-18, image → material features
  4. Multimodal Fusion – bilinear fusion of echo/image/mat
  5. Attention Net  – per-pixel α to combine echo & image

Final depth:  D = α ⊙ D_echo + (1 − α) ⊙ D_image

Image input is set to zeros (deactivated but architecturally present).

Reference impl: https://github.com/krantiparida/beyond-image-to-depth
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


# ── helpers ────────────────────────────────────────────────────

def _unet_conv(in_c, out_c, norm_layer=nn.BatchNorm2d):
    """Downsample: Conv(k4,s2,p1) + BN + LeakyReLU."""
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1),
        norm_layer(out_c),
        nn.LeakyReLU(0.2, True),
    )


def _unet_upconv(in_c, out_c, outermost=False, norm_layer=nn.BatchNorm2d):
    """Upsample: ConvTranspose(k4,s2,p1) + BN + ReLU  (Sigmoid if outermost)."""
    if outermost:
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )
    return nn.Sequential(
        nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1),
        norm_layer(out_c),
        nn.ReLU(True),
    )


def _create_conv(in_c, out_c, kernel, padding, batch_norm=True, relu=True, stride=1):
    layers = [nn.Conv2d(in_c, out_c, kernel, stride=stride, padding=padding)]
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_c))
    if relu:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def _weights_init(m):
    name = m.__class__.__name__
    if name.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif name.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif name.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)


def _conv_output_dim(dimension, kernel_size, stride):
    """Spatial dims after conv with no padding."""
    dim = np.asarray(dimension, dtype=np.float64)
    ks = np.asarray(kernel_size, dtype=np.float64)
    st = np.asarray(stride, dtype=np.float64)
    return np.floor((dim - ks) / st + 1).astype(int)


# ── Sub-networks ───────────────────────────────────────────────

class EchoSubNet(nn.Module):
    """Echo encoder-decoder (Sec 3.2)."""

    def __init__(self, audio_shape, conv1x1_dim=8, bottleneck_dim=512,
                 output_nc=1, output_size=(256, 512)):
        super().__init__()
        self.output_size = output_size
        n_input = audio_shape[0]
        kernel_sizes = [(8, 8), (4, 4), (3, 3)]
        strides = [(4, 4), (2, 2), (1, 1)]

        dims = np.array(audio_shape[1:], dtype=np.float64)
        for ks, st in zip(kernel_sizes, strides):
            dims = _conv_output_dim(dims, ks, st)

        self.encoder = nn.Sequential(
            _create_conv(n_input, 32, kernel_sizes[0], 0, stride=strides[0]),
            _create_conv(32, 64, kernel_sizes[1], 0, stride=strides[1]),
            _create_conv(64, conv1x1_dim, kernel_sizes[2], 0, stride=strides[2]),
        )
        flatten_dim = int(conv1x1_dim * dims[0] * dims[1])
        self.bottleneck = _create_conv(flatten_dim, bottleneck_dim, 1, 0)

        self.decoder = nn.Sequential(
            _unet_upconv(bottleneck_dim, 512),
            _unet_upconv(512, 256),
            _unet_upconv(256, 128),
            _unet_upconv(128, 64),
            _unet_upconv(64, 32),
            _unet_upconv(32, 16),
            _unet_upconv(16, output_nc, outermost=True),
        )

    def forward(self, x):
        feat = self.encoder(x)
        feat = feat.view(feat.shape[0], -1, 1, 1)
        feat = self.bottleneck(feat)
        depth = self.decoder(feat)
        if depth.shape[2:] != self.output_size:
            depth = F.interpolate(depth, size=self.output_size,
                                  mode='bilinear', align_corners=False)
        return depth, feat


class VisualSubNet(nn.Module):
    """Visual UNet encoder-decoder (Sec 3.3)."""

    def __init__(self, ngf=64, input_nc=3, output_nc=1):
        super().__init__()
        self.conv1 = _unet_conv(input_nc, ngf)
        self.conv2 = _unet_conv(ngf, ngf * 2)
        self.conv3 = _unet_conv(ngf * 2, ngf * 4)
        self.conv4 = _unet_conv(ngf * 4, ngf * 8)
        self.conv5 = _unet_conv(ngf * 8, ngf * 8)
        self.up1 = _unet_upconv(512, ngf * 8)
        self.up2 = _unet_upconv(ngf * 16, ngf * 4)
        self.up3 = _unet_upconv(ngf * 8, ngf * 2)
        self.up4 = _unet_upconv(ngf * 4, ngf)
        self.up5 = _unet_upconv(ngf * 2, output_nc, outermost=True)

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        u1 = self.up1(c5)
        u2 = self.up2(torch.cat([u1, c4], dim=1))
        u3 = self.up3(torch.cat([u2, c3], dim=1))
        u4 = self.up4(torch.cat([u3, c2], dim=1))
        depth = self.up5(torch.cat([u4, c1], dim=1))
        return depth, c5


class MaterialSubNet(nn.Module):
    """Material property network (Sec 3.4)."""

    def __init__(self, n_class=23):
        super().__init__()
        backbone = tv_models.resnet18(weights=None)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, n_class)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feat = self.layer4(x)
        cls = self.fc(self.pool(feat).flatten(1))
        return cls, feat


class MultimodalFusion(nn.Module):
    """Bilinear fusion (Sec 3.5)."""

    def __init__(self, feat_dim=512):
        super().__init__()
        self.bilinear_img = nn.Bilinear(feat_dim, feat_dim, feat_dim)
        self.bilinear_mat = nn.Bilinear(feat_dim, feat_dim, feat_dim)

    def forward(self, img_feat, echo_feat, mat_feat):
        B, C, H, W = img_feat.shape
        img_f = img_feat.permute(0, 2, 3, 1).contiguous()
        echo_f = echo_feat.permute(0, 2, 3, 1).contiguous()
        mat_f = mat_feat.permute(0, 2, 3, 1).contiguous()
        f_img = self.bilinear_img(img_f, echo_f).permute(0, 3, 1, 2).contiguous()
        f_mat = self.bilinear_mat(mat_f, echo_f).permute(0, 3, 1, 2).contiguous()
        return torch.cat([f_img, f_mat], dim=1)


class AttentionSubNet(nn.Module):
    """Attention prediction network (Sec 3.6)."""

    def __init__(self, input_nc=1024):
        super().__init__()
        self.up1 = _unet_upconv(input_nc, 512)
        self.up2 = _unet_upconv(512, 256)
        self.up3 = _unet_upconv(256, 128)
        self.up4 = _unet_upconv(128, 64)
        self.up5 = _unet_upconv(64, 1, outermost=True)

    def forward(self, fused):
        x = self.up1(fused)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        alpha = self.up5(x)
        return alpha


# ── Full model ─────────────────────────────────────────────────

class EchoNet(nn.Module):
    """Full 'Beyond Image to Depth' model with image branch deactivated.

    Interface matches the baseline: input (B,2,H,W) audio → output (B,1,H,W) depth.
    """

    def __init__(self, cfg, input_nc=2, output_nc=1,
                 conv1x1_dim=8, bottleneck_dim=64):
        super().__init__()
        self.output_size = tuple(int(s) for s in cfg.dataset.images_size)
        self.max_depth = getattr(cfg.dataset, 'max_depth', 10.0)
        depth_norm = getattr(cfg.dataset, 'depth_norm', True)
        audio_shape = (input_nc, *self.output_size)

        self.echo_net = EchoSubNet(
            audio_shape=audio_shape, conv1x1_dim=conv1x1_dim,
            bottleneck_dim=bottleneck_dim, output_nc=output_nc,
            output_size=self.output_size)
        self.visual_net = VisualSubNet(ngf=64, input_nc=3, output_nc=output_nc)
        self.material_net = MaterialSubNet(n_class=23)
        self.proj_img = nn.Conv2d(512, bottleneck_dim, 1)
        self.proj_mat = nn.Conv2d(512, bottleneck_dim, 1)
        self.fusion = MultimodalFusion(feat_dim=bottleneck_dim)
        self.attention_net = AttentionSubNet(input_nc=bottleneck_dim * 2)
        self.apply(_weights_init)

    def forward(self, audio):
        B = audio.shape[0]
        device = audio.device
        zero_img = torch.zeros(B, 3, *self.output_size, device=device)

        echo_depth, echo_feat = self.echo_net(audio)
        img_depth, img_feat = self.visual_net(zero_img)
        _, mat_feat = self.material_net(zero_img)

        img_feat = self.proj_img(img_feat)
        mat_feat = self.proj_mat(mat_feat)
        echo_feat_spatial = echo_feat.expand(-1, -1, img_feat.shape[2], img_feat.shape[3])
        fused = self.fusion(img_feat, echo_feat_spatial, mat_feat)
        alpha = self.attention_net(fused)

        target = self.output_size
        if alpha.shape[2:] != target:
            alpha = F.interpolate(alpha, size=target, mode='bilinear', align_corners=False)
        if img_depth.shape[2:] != target:
            img_depth = F.interpolate(img_depth, size=target, mode='bilinear', align_corners=False)

        depth = alpha * echo_depth + (1 - alpha) * img_depth
        return depth
