"""v5 — SH-conditioned cross-attention on the depth decoder input.

Rationale
---------
v3 injects FOA info via a global FiLM (scalar modulation shared over
tokens), v4 mixes information across ViT depths. v5 instead gives the
depth decoder *spatially-addressable* ambisonic conditioning:

  final ViT tokens ──► SHHead ──► pred_sh (B, sh_dim)
                                    │
                                    ▼
                       SHToTokens (K learnable queries × sh_dim)
                                    │
                                    ▼
                      patch_tokens ──► CrossAttn(Q=patches, KV=sh tokens)
                                    │
                                    ▼
                                DepthDecoder

Each of the K SH-derived tokens can represent a directional "slot"
(think: front/back/left/right/up/down for FOA, extended for HOA), and
every patch of the depth-decoder input queries these slots. Because
the cross-attention is per-patch, the directional cue can influence
depth differently in different regions of the ERP panorama, which is
exactly what a fixed FiLM cannot do.

Loss routing: v1 aux-SH-head path (``_train_step_foa_0415``).
"""

import torch
import torch.nn as nn

from .pretrained_vit_foa import ViTFOABackbone, SHHead


class SHCrossAttnBlock(nn.Module):
    """LayerNorm -> MultiHeadCrossAttention -> residual + FFN."""

    def __init__(self, embed_dim: int, num_heads: int = 8, mlp_ratio: float = 2.0,
                 dropout: float = 0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, q_tokens, kv_tokens):
        q = self.norm_q(q_tokens)
        kv = self.norm_kv(kv_tokens)
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        q_tokens = q_tokens + attn_out
        q_tokens = q_tokens + self.mlp(self.norm2(q_tokens))
        return q_tokens


class PretrainedViTFOAV5(ViTFOABackbone):

    DEFAULT_SH_DIM = 4
    DEFAULT_HEAD_HIDDEN = 256
    DEFAULT_N_SLOTS = 8              # number of SH conditioning tokens
    DEFAULT_XATTN_HEADS = 8

    def __init__(self, cfg, input_nc=2, pretrained=True, freeze_encoder=False,
                 sh_dim: int = None, head_hidden: int = None,
                 n_slots: int = None, num_heads: int = None, **_unused):
        super().__init__(cfg, input_nc=input_nc, pretrained=pretrained,
                         freeze_encoder=freeze_encoder)
        self.sh_dim = int(sh_dim if sh_dim is not None else self.DEFAULT_SH_DIM)
        hidden = int(head_hidden if head_hidden is not None
                     else self.DEFAULT_HEAD_HIDDEN)
        self.n_slots = int(n_slots if n_slots is not None else self.DEFAULT_N_SLOTS)
        num_heads = int(num_heads if num_heads is not None
                        else self.DEFAULT_XATTN_HEADS)

        # SH head on final tokens (same as v1)
        self.sh_head = SHHead(feat_dim=2 * self.EMBED_DIM, sh_dim=self.sh_dim,
                              hidden=hidden)

        # Project each SH coefficient to a slot token, then expand to n_slots
        # learnable bases so every SH coefficient lands in every slot.
        self.slot_bias = nn.Parameter(torch.zeros(self.n_slots, self.EMBED_DIM))
        self.sh_to_slots = nn.Sequential(
            nn.LayerNorm(self.sh_dim),
            nn.Linear(self.sh_dim, self.n_slots * self.EMBED_DIM),
        )
        self.xattn = SHCrossAttnBlock(self.EMBED_DIM, num_heads=num_heads)

    def forward(self, x, **_unused):
        orig_h, orig_w = x.shape[2], x.shape[3]
        tokens = self._encode(x)                          # (B, 1+N, D)
        cls_feat = tokens[:, 0]
        patch_tokens = tokens[:, 1:]                      # (B, N, D)
        patch_mean = patch_tokens.mean(dim=1)

        # SH prediction from final tokens
        pred_sh = self.sh_head(torch.cat([cls_feat, patch_mean], dim=1))

        # SH -> K conditioning tokens
        B = pred_sh.shape[0]
        slots = self.sh_to_slots(pred_sh).view(B, self.n_slots, self.EMBED_DIM)
        slots = slots + self.slot_bias.unsqueeze(0)       # (B, K, D)

        # Patch tokens cross-attend to SH slots
        cond_patches = self.xattn(patch_tokens, slots)    # (B, N, D)

        pred_depth = self._decode_depth(cond_patches, orig_h, orig_w)
        return {"pred_depth": pred_depth, "pred_sh": pred_sh}
