"""Pretrained ViT-B/16 with auxiliary SH head for audio-to-depth.

Defines a shared backbone (``ViTFOABackbone``) used by the v1..v5 family
and the concrete v1 variant (``PretrainedViTFOA``):

  v1 — Aux SH head off final tokens, no decoder injection (this file).
  v2 — + histogram alignment with energy-map SH projection.
  v3 — Early-layer SH tap feeds FiLM into the depth decoder.
  v4 — Multi-scale SH aggregation from 4 tapped layers (DPT-style).
  v5 — SH cross-attention tokens conditioning the depth decoder.

Pipeline (v1)
-------------
  x (B, 2, H, W) spectrogram
      -> input_adapter (Conv2d 2->3, 1x1)      pseudo-RGB
      -> patch_embed (ViT conv_proj)           tokens (B, N, 768)
      -> prepend CLS + ViT encoder (pos emb)   tokens (B, 1+N, 768)
      -> patch_tokens -> DepthDecoder          (B, 1, H, W)
      -> [cls, mean(patch)] -> SHHead          (B, sh_dim)
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models
from torchvision.models import ViT_B_16_Weights

from .pretrained_vit import DepthDecoder, _interpolate_pos_embed


class SHHead(nn.Module):
    """LayerNorm -> Linear -> GELU -> Linear head producing SH coefficients."""

    def __init__(self, feat_dim: int, sh_dim: int, hidden: int = 256):
        super().__init__()
        self.norm = nn.LayerNorm(feat_dim)
        self.fc1 = nn.Linear(feat_dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, sh_dim)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(self.norm(feat))))


class ViTFOABackbone(nn.Module):
    """Shared ViT-B/16 backbone + depth decoder for the pretrained_vit_foa family.

    Provides:
      * input_adapter (2 -> 3 channels)
      * patch_embed, cls_token, encoder (ViT-B/16) with pos-embed interpolated
        to our grid
      * DepthDecoder (from pretrained_vit.py)
      * ``_encode(x)``          — standard ViT forward, returns final tokens
      * ``_encode_with_taps``   — manual block loop capturing intermediate tokens
      * ``_decode_depth``       — patch-token -> dense depth map

    Subclasses add the SH branch / conditioning / loss hooks.
    """

    EMBED_DIM = 768
    PATCH_SIZE = 16
    NUM_LAYERS = 12

    def __init__(self, cfg, input_nc=2, pretrained=True, freeze_encoder=False):
        super().__init__()
        self.target_h = int(cfg.dataset.images_size[0])
        self.target_w = int(cfg.dataset.images_size[1])
        depth_norm = getattr(cfg.dataset, 'depth_norm', True)

        self.input_adapter = nn.Conv2d(input_nc, 3, kernel_size=1)

        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        vit = tv_models.vit_b_16(weights=weights)
        self.patch_embed = vit.conv_proj
        self.cls_token = vit.class_token
        self.encoder = vit.encoder

        self.grid_h = self.target_h // self.PATCH_SIZE
        self.grid_w = self.target_w // self.PATCH_SIZE

        old_pos = vit.encoder.pos_embedding.data
        new_pos = _interpolate_pos_embed(old_pos, (14, 14),
                                         (self.grid_h, self.grid_w))
        self.encoder.pos_embedding = nn.Parameter(new_pos)

        if freeze_encoder:
            for p in self.patch_embed.parameters():
                p.requires_grad = False
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.cls_token.requires_grad = False

        self.decoder = DepthDecoder(self.EMBED_DIM, self.grid_h, self.grid_w,
                                    depth_norm=depth_norm)

    def _embed(self, x):
        """(B, 2, H, W) -> (B, 1+N, D) tokens ready for the encoder (no pos-emb yet)."""
        x = self.input_adapter(x)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)             # (B, N, D)
        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        return torch.cat([cls, x], dim=1)            # (B, 1+N, D)

    def _encode(self, x):
        """Standard ViT encoder forward. Returns final tokens (B, 1+N, D)."""
        tokens = self._embed(x)
        return self.encoder(tokens)

    def _encode_with_taps(self, x, tap_indices):
        """Run the ViT encoder block-by-block, capturing intermediate tokens.

        tap_indices : iterable of int in [0, NUM_LAYERS-1].
        Returns ``(final_tokens, taps)`` where ``taps[i]`` is the output of
        block ``i`` (pre-final-LN; for the last block use ``final_tokens``).
        """
        tap_set = set(int(i) for i in tap_indices)
        tokens = self._embed(x)
        tokens = tokens + self.encoder.pos_embedding
        tokens = self.encoder.dropout(tokens)
        taps = {}
        for i, block in enumerate(self.encoder.layers):
            tokens = block(tokens)
            if i in tap_set:
                taps[i] = tokens
        final = self.encoder.ln(tokens)
        return final, taps

    def _decode_depth(self, patch_tokens, target_h, target_w):
        return self.decoder(patch_tokens, target_h, target_w)


class PretrainedViTFOA(ViTFOABackbone):
    """v1 — Aux SH head off final tokens, no decoder injection.

    Output dict: ``{pred_depth, pred_sh}``.
    Trained via the ``_train_step_foa_0415`` path:
        L = L_berhu + lambda_si * L_silog + lambda_sh * L1(pred_sh[:, :4], gt_foa)
    """

    DEFAULT_SH_DIM = 4
    DEFAULT_HEAD_HIDDEN = 256

    def __init__(self, cfg, input_nc=2, pretrained=True, freeze_encoder=False,
                 sh_dim: int = None, head_hidden: int = None, **_unused):
        super().__init__(cfg, input_nc=input_nc, pretrained=pretrained,
                         freeze_encoder=freeze_encoder)
        self.sh_dim = int(sh_dim if sh_dim is not None else self.DEFAULT_SH_DIM)
        hidden = int(head_hidden if head_hidden is not None
                     else self.DEFAULT_HEAD_HIDDEN)
        self.sh_head = SHHead(feat_dim=2 * self.EMBED_DIM, sh_dim=self.sh_dim,
                              hidden=hidden)

    def forward(self, x, **_unused):
        orig_h, orig_w = x.shape[2], x.shape[3]
        tokens = self._encode(x)                         # (B, 1+N, D)
        cls_feat = tokens[:, 0]                          # (B, D)
        patch_tokens = tokens[:, 1:]                     # (B, N, D)
        patch_mean = patch_tokens.mean(dim=1)            # (B, D)
        pred_sh = self.sh_head(torch.cat([cls_feat, patch_mean], dim=1))
        pred_depth = self._decode_depth(patch_tokens, orig_h, orig_w)
        return {"pred_depth": pred_depth, "pred_sh": pred_sh}
