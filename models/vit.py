"""Vision Transformer (ViT) for audio-to-depth estimation.

A ViT encoder with a convolutional decoder for dense depth prediction.
Input: (B, 2, H, W) binaural spectrogram -> Output: (B, 1, H, W) depth map.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PatchEmbed(nn.Module):
    """2D image to patch embedding."""

    def __init__(self, img_size=(256, 512), patch_size=16, in_chans=2, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x


class Attention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """Feed-forward network."""

    def __init__(self, dim, mlp_ratio=4., drop=0.):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block with pre-norm."""

    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ConvDecoder(nn.Module):
    """Progressive upsampling decoder from patch tokens to dense depth."""

    def __init__(self, embed_dim, img_size=(256, 512), patch_size=16,
                 depth_norm=True):
        super().__init__()
        self.grid_h = img_size[0] // patch_size
        self.grid_w = img_size[1] // patch_size
        self.depth_norm = depth_norm

        # Progressive upsampling: 4 stages, each 2x
        # From (embed_dim, grid_h, grid_w) to (1, H, W)
        ch = embed_dim
        layers = []
        for out_ch in [256, 128, 64, 32]:
            layers.append(nn.Sequential(
                nn.ConvTranspose2d(ch, out_ch, 4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ))
            ch = out_ch
        self.up_layers = nn.ModuleList(layers)

        self.head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, tokens, orig_h, orig_w):
        # tokens: (B, N, C) -> reshape to spatial
        x = rearrange(tokens, 'b (h w) c -> b c h w',
                       h=self.grid_h, w=self.grid_w)

        for layer in self.up_layers:
            x = layer(x)

        x = self.head(x)
        if self.depth_norm:
            x = torch.sigmoid(x)

        # Resize to exact original dimensions if needed
        if x.shape[2] != orig_h or x.shape[3] != orig_w:
            x = F.interpolate(x, size=(orig_h, orig_w), mode='nearest')
        return x


class AudioDepthViT(nn.Module):
    """ViT-based audio-to-depth estimation.

    Input:  (B, 2, H, W) binaural spectrogram
    Output: (B, 1, H, W) predicted depth map

    Architecture:
        PatchEmbed -> [TransformerBlock x depth] -> ConvDecoder -> depth
    """

    def __init__(self, cfg, img_size=(256, 512), patch_size=16, in_chans=2,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth_norm = cfg.dataset.depth_norm

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, drop=drop_rate,
                             attn_drop=attn_drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.decoder = ConvDecoder(embed_dim, img_size, patch_size,
                                   depth_norm=self.depth_norm)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        orig_h, orig_w = x.shape[2], x.shape[3]
        B = x.shape[0]

        # Patch embed
        x = self.patch_embed(x)

        # Prepend CLS token + add positional embedding
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)

        # Transformer encoder
        x = self.blocks(x)
        x = self.norm(x)

        # Remove CLS token, decode to depth
        patch_tokens = x[:, 1:, :]
        depth = self.decoder(patch_tokens, orig_h, orig_w)
        return depth
