"""Pretrained ViT-B/16 adapted for audio-to-depth estimation.

Uses a learnable 1×1 conv to project 2-channel binaural spectrogram
to 3-channel pseudo-RGB, then feeds through a pretrained ViT-B/16
encoder with interpolated positional embeddings.  A lightweight
progressive ConvTranspose decoder maps patch tokens back to a dense
depth map.

Positional embeddings are interpolated at init so the encoder
natively handles (256, 512) input without runtime resizing.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from torchvision.models import ViT_B_16_Weights


class DepthDecoder(nn.Module):
    """Progressive upsampling from patch-token grid to dense depth."""

    def __init__(self, embed_dim, grid_h, grid_w, depth_norm=True):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.depth_norm = depth_norm

        ch = embed_dim
        layers = []
        for out_ch in [256, 128, 64, 32]:
            layers.append(nn.Sequential(
                nn.ConvTranspose2d(ch, out_ch, 4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(True),
            ))
            ch = out_ch
        self.up = nn.ModuleList(layers)
        self.head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, tokens, target_h, target_w):
        B, N, C = tokens.shape
        x = tokens.transpose(1, 2).reshape(B, C, self.grid_h, self.grid_w)
        for layer in self.up:
            x = layer(x)
        x = self.head(x)
        if self.depth_norm:
            x = torch.sigmoid(x)
        if x.shape[2] != target_h or x.shape[3] != target_w:
            x = F.interpolate(x, (target_h, target_w), mode='bilinear',
                              align_corners=False)
        return x


def _interpolate_pos_embed(pos_embed, old_grid, new_grid):
    """Interpolate ViT positional embeddings to a new grid size.

    Parameters
    ----------
    pos_embed : Tensor (1, 1+old_H*old_W, D)
        Original pos embeddings (CLS + patch tokens).
    old_grid : (int, int)
        Original (H, W) grid.
    new_grid : (int, int)
        Target (H, W) grid.

    Returns
    -------
    Tensor (1, 1+new_H*new_W, D)
    """
    cls_token = pos_embed[:, :1, :]          # (1, 1, D)
    patch_pos = pos_embed[:, 1:, :]          # (1, old_H*old_W, D)
    D = patch_pos.shape[-1]
    oH, oW = old_grid
    nH, nW = new_grid
    patch_pos = patch_pos.reshape(1, oH, oW, D).permute(0, 3, 1, 2)
    patch_pos = F.interpolate(patch_pos, size=(nH, nW), mode='bicubic',
                              align_corners=False)
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, nH * nW, D)
    return torch.cat([cls_token, patch_pos], dim=1)


class PretrainedViT(nn.Module):
    """Pretrained ViT-B/16 for audio-to-depth.

    Pipeline:
      1. Input adapter — Conv2d(2, 3, 1×1): spectrogram → pseudo-RGB
      2. ViT-B/16 encoder (ImageNet pretrained) with interpolated pos embeds
      3. Progressive ConvTranspose decoder → depth map

    Parameters
    ----------
    cfg : SimpleNamespace
        Needs ``cfg.dataset.images_size`` and ``cfg.dataset.depth_norm``.
    input_nc : int
        Audio channels (2).
    pretrained : bool
        Load ImageNet weights.
    freeze_encoder : bool
        Freeze ViT encoder weights (fine-tune decoder only).
    """

    def __init__(self, cfg, input_nc=2, pretrained=True, freeze_encoder=False):
        super().__init__()
        self.target_h = int(cfg.dataset.images_size[0])
        self.target_w = int(cfg.dataset.images_size[1])
        depth_norm = getattr(cfg.dataset, 'depth_norm', True)
        patch_size = 16
        embed_dim = 768

        # (1) Input adapter: 2ch spectrogram → 3ch pseudo-RGB
        self.input_adapter = nn.Conv2d(input_nc, 3, kernel_size=1)

        # (2) Pretrained ViT-B/16
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        vit = tv_models.vit_b_16(weights=weights)

        # Keep only the encoder parts
        self.patch_embed = vit.conv_proj          # Conv2d(3, 768, 16, 16)
        self.cls_token = vit.class_token          # (1, 1, 768)
        self.encoder = vit.encoder                # Encoder (pos_embed + layers)

        # Grid for our input resolution
        self.grid_h = self.target_h // patch_size  # 16
        self.grid_w = self.target_w // patch_size  # 32
        n_patches = self.grid_h * self.grid_w

        # Interpolate positional embeddings from 14×14 → our grid
        # torchvision pos_embedding: (1, 1+14*14, 768) → (1, 1+grid_h*grid_w, 768)
        old_pos = vit.encoder.pos_embedding.data   # (1, 197, 768)
        new_pos = _interpolate_pos_embed(old_pos, (14, 14),
                                         (self.grid_h, self.grid_w))
        self.encoder.pos_embedding = nn.Parameter(new_pos)

        if freeze_encoder:
            for p in self.patch_embed.parameters():
                p.requires_grad = False
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.cls_token.requires_grad = False

        # (3) Decoder
        self.decoder = DepthDecoder(embed_dim, self.grid_h, self.grid_w,
                                    depth_norm=depth_norm)

    def forward(self, x):
        """
        x : (B, 2, H, W) binaural spectrogram
        Returns: (B, 1, H, W) depth
        """
        orig_h, orig_w = x.shape[2], x.shape[3]

        # Adapt channels
        x = self.input_adapter(x)                  # (B, 3, H, W)

        # Patch embed: (B, 3, H, W) → (B, n_patches, 768)
        x = self.patch_embed(x)                    # (B, 768, grid_h, grid_w)
        x = x.flatten(2).transpose(1, 2)           # (B, n_patches, 768)

        # Prepend CLS token (torchvision Encoder expects CLS already prepended)
        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)     # (B, 1, 768)
        x = torch.cat([cls, x], dim=1)             # (B, 1+n_patches, 768)

        # Encoder: adds pos_embedding, runs transformer layers
        x = self.encoder(x)                        # (B, 1+n_patches, 768)

        # Drop CLS token, decode
        patch_tokens = x[:, 1:, :]
        depth = self.decoder(patch_tokens, orig_h, orig_w)
        return depth
