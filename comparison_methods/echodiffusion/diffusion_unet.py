"""Standalone diffusion UNet for EchoDiffusion feature extraction.

Implements the UNet from Stable Diffusion (OpenAI architecture) with:
- Sinusoidal timestep embeddings
- ResBlocks with timestep conditioning
- SpatialTransformer blocks with cross-attention for context conditioning
- Hierarchical feature extraction (no output head)

Config matches v1-inference.yaml:
  in_channels=512, model_channels=32, channel_mult=[1,2,4,4],
  num_res_blocks=2, attention_resolutions=[4,2,1],
  num_heads=8, context_dim=768, transformer_depth=1
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ── Utilities ────────────────────────────────────────────────

def timestep_embedding(timesteps, dim, max_period=10000):
    """Sinusoidal timestep embedding."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32,
                                                             device=timesteps.device) / half)
    args = timesteps[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


# ── Attention ────────────────────────────────────────────────

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim or query_dim
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def forward(self, x, context=None):
        h = self.heads
        context = context if context is not None else x

        q = rearrange(self.to_q(x), 'b n (h d) -> (b h) n d', h=h)
        k = rearrange(self.to_k(context), 'b n (h d) -> (b h) n d', h=h)
        v = rearrange(self.to_v(context), 'b n (h d) -> (b h) n d', h=h)

        attn = torch.bmm(q, k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.bmm(attn, v)
        return self.to_out(rearrange(out, '(b h) n d -> b n (h d)', h=h))


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, context_dim=None, dropout=0.0):
        super().__init__()
        self.attn1 = CrossAttention(dim, heads=n_heads, dim_head=d_head, dropout=dropout)
        self.attn2 = CrossAttention(dim, context_dim=context_dim, heads=n_heads,
                                    dim_head=d_head, dropout=dropout)
        self.ff = FeedForward(dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """Applies transformer blocks to 2D feature maps with optional cross-attention context."""

    def __init__(self, in_channels, n_heads, d_head, depth=1, context_dim=None, dropout=0.0):
        super().__init__()
        inner_dim = n_heads * d_head
        self.norm = nn.GroupNorm(32, in_channels, eps=1e-6)
        self.proj_in = nn.Conv2d(in_channels, inner_dim, 1)
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(inner_dim, n_heads, d_head, context_dim=context_dim,
                                 dropout=dropout)
            for _ in range(depth)
        ])
        self.proj_out = zero_module(nn.Conv2d(inner_dim, in_channels, 1))

    def forward(self, x, context=None):
        b, c, h, w = x.shape
        x_in = x
        x = self.proj_in(self.norm(x))
        x = rearrange(x, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return self.proj_out(x) + x_in


# ── ResBlock ─────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, channels, emb_channels, out_channels=None, dropout=0.0):
        super().__init__()
        out_channels = out_channels or channels
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, out_channels),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            zero_module(nn.Conv2d(out_channels, out_channels, 3, padding=1)),
        )
        self.skip_connection = (nn.Conv2d(channels, out_channels, 1)
                                if channels != out_channels else nn.Identity())

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb)[:, :, None, None]
        h = h + emb_out
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        return self.conv(x)


# ── TimestepBlock wrapper ────────────────────────────────────

class TimestepSequential(nn.Sequential):
    """Sequential that passes timestep embedding and context to children that need them."""

    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context=context)
            else:
                x = layer(x)
        return x


# ── UNet Model ───────────────────────────────────────────────

class DiffusionUNet(nn.Module):
    """UNet with cross-attention, matching the SD architecture for EchoDiffusion.

    Used as a feature extractor: returns hierarchical features from output_blocks
    at indices [1, 4, 7] plus the final hidden state.
    """

    def __init__(self, in_channels=512, out_channels=4, model_channels=32,
                 channel_mult=(1, 2, 4, 4), num_res_blocks=2,
                 attention_resolutions=(4, 2, 1), num_heads=8,
                 context_dim=768, transformer_depth=1, dropout=0.0):
        super().__init__()
        self.model_channels = model_channels
        time_embed_dim = model_channels * 4
        self.dtype = torch.float32

        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Encoder
        self.input_blocks = nn.ModuleList()
        ch = model_channels
        self.input_blocks.append(TimestepSequential(
            nn.Conv2d(in_channels, ch, 3, padding=1)))
        input_block_chans = [ch]
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                out_ch = model_channels * mult
                layers = [ResBlock(ch, time_embed_dim, out_ch, dropout=dropout)]
                ch = out_ch
                if ds in attention_resolutions:
                    dim_head = ch // num_heads
                    layers.append(SpatialTransformer(
                        ch, num_heads, dim_head, depth=transformer_depth,
                        context_dim=context_dim, dropout=dropout))
                self.input_blocks.append(TimestepSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(TimestepSequential(Downsample(ch)))
                input_block_chans.append(ch)
                ds *= 2

        # Middle
        dim_head = ch // num_heads
        self.middle_block = TimestepSequential(
            ResBlock(ch, time_embed_dim, ch, dropout=dropout),
            SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth,
                               context_dim=context_dim, dropout=dropout),
            ResBlock(ch, time_embed_dim, ch, dropout=dropout),
        )

        # Decoder
        self.output_blocks = nn.ModuleList()
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                out_ch = model_channels * mult
                layers = [ResBlock(ch + ich, time_embed_dim, out_ch, dropout=dropout)]
                ch = out_ch
                if ds in attention_resolutions:
                    dim_head = ch // num_heads
                    layers.append(SpatialTransformer(
                        ch, num_heads, dim_head, depth=transformer_depth,
                        context_dim=context_dim, dropout=dropout))
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch))
                    ds //= 2
                self.output_blocks.append(TimestepSequential(*layers))

        self.num_classes = None

    def forward(self, x, timesteps, context=None):
        """Returns hierarchical features [finest, ..., coarsest] (4 items)."""
        t_emb = timestep_embedding(timesteps, self.model_channels)
        emb = self.time_embed(t_emb)

        hs = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)

        h = self.middle_block(h, emb, context)

        out_list = []
        for i_out, module in enumerate(self.output_blocks):
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
            if i_out in [1, 4, 7]:
                out_list.append(h)

        out_list.append(h)
        return out_list[::-1]  # finest first
