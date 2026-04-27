"""v6 — ViT ports of the n3_0419 breakthrough variants (exp216-219).

Three classes share the ViTFOABackbone from pretrained_vit_foa.py and
port the successful UNet-side N3 mechanisms into the ViT backbone:

  v6_eattn      — aux SH head + predicted energy map from patch tokens
                  (ViT analogue of n3_energy_attn / exp172 family).
                  Output: {pred_depth, pred_sh, pred_energy}.

  v6_mssh       — multi-scale SH heads tapped from 4 ViT blocks + fused
                  prediction (ViT analogue of n3_multiscale_sh / exp171).
                  Output: {pred_depth, pred_sh, pred_sh_scales}.

  v6_oracle_nc3 — v1-style ViT with input_nc=3 (binaural + GT energy map).
                  Oracle upper bound for ViT backbone (ViT analogue of
                  foa_oracle_nc3 / exp180).
                  Output: {pred_depth, pred_sh}.

All three route through the standard foa_0415 training path
(`_train_step_foa_0415`) except v6_oracle_nc3 which routes through the
oracle path (`_train_step_oracle`). Registration lives in
utils/train_utils.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pretrained_vit_foa import ViTFOABackbone, SHHead


class ViTEnergyHead(nn.Module):
    """Patch-token grid -> (B, 1, H, W) energy map, sigmoid-bounded.

    Mirrors the DepthDecoder structure but outputs a scalar channel. The
    final Sigmoid keeps the map in [0, 1] so it can be used either as a
    multiplicative attention gate or supervised against a normalized GT
    energy map.
    """

    def __init__(self, embed_dim: int, grid_h: int, grid_w: int):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        ch = embed_dim
        layers = []
        for out_ch in (256, 128, 64):
            layers.append(nn.Sequential(
                nn.ConvTranspose2d(ch, out_ch, 4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(True),
            ))
            ch = out_ch
        self.up = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Conv2d(64, 16, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, patch_tokens: torch.Tensor,
                target_h: int, target_w: int) -> torch.Tensor:
        B, N, C = patch_tokens.shape
        x = patch_tokens.transpose(1, 2).reshape(B, C, self.grid_h, self.grid_w)
        x = self.up(x)
        x = self.head(x)
        if x.shape[2] != target_h or x.shape[3] != target_w:
            x = F.interpolate(x, size=(target_h, target_w),
                              mode='bilinear', align_corners=False)
        return x


class PretrainedViTFOAV6EAttn(ViTFOABackbone):
    """v6_eattn — ViT + predicted energy map head (exp216).

    Pipeline:
      x -> ViT encoder -> final tokens (B, 1+N, D)
      - aux SH head on [cls, mean(patch)]            -> pred_sh
      - depth decoder on patch tokens                -> pred_depth
      - ViTEnergyHead on patch tokens                -> pred_energy

    The energy map is returned for optional supervision (via
    ``cfg.model.lambda_energy`` > 0 in _train_step_foa_0415); it does not
    modulate the decoder directly (no in-place attention on patch tokens
    — the ViT decoder is already light and the dominant win in UNet came
    from the supervision gradient, not the multiplicative gate).
    """

    DEFAULT_SH_DIM = 4
    DEFAULT_HEAD_HIDDEN = 256

    def __init__(self, cfg, input_nc: int = 2, pretrained: bool = True,
                 freeze_encoder: bool = False,
                 sh_dim: int = None, head_hidden: int = None, **_unused):
        super().__init__(cfg, input_nc=input_nc, pretrained=pretrained,
                         freeze_encoder=freeze_encoder)
        self.sh_dim = int(sh_dim if sh_dim is not None else self.DEFAULT_SH_DIM)
        hidden = int(head_hidden if head_hidden is not None
                     else self.DEFAULT_HEAD_HIDDEN)

        self.sh_head = SHHead(feat_dim=2 * self.EMBED_DIM, sh_dim=self.sh_dim,
                              hidden=hidden)
        self.energy_head = ViTEnergyHead(self.EMBED_DIM,
                                         self.grid_h, self.grid_w)

    def forward(self, x, **_unused):
        orig_h, orig_w = x.shape[2], x.shape[3]
        tokens = self._encode(x)                           # (B, 1+N, D)
        cls_feat = tokens[:, 0]
        patch_tokens = tokens[:, 1:]
        patch_mean = patch_tokens.mean(dim=1)
        pred_sh = self.sh_head(torch.cat([cls_feat, patch_mean], dim=1))
        pred_depth = self._decode_depth(patch_tokens, orig_h, orig_w)
        pred_energy = self.energy_head(patch_tokens, orig_h, orig_w)
        return {
            "pred_depth": pred_depth,
            "pred_sh": pred_sh,
            "pred_energy": pred_energy,
        }


class PretrainedViTFOAV6MSSH(ViTFOABackbone):
    """v6_mssh — ViT + multi-scale SH taps (exp217).

    SH heads tap four ViT layers (default [2, 5, 8, 11]), each producing
    a per-scale SH prediction from [cls_tap, mean_patch_tap]. A learnable
    linear mix fuses the four into ``pred_sh``; per-scale predictions are
    returned for downstream multi-scale loss (currently summed implicitly
    via the fused pred_sh; exposed here for optional explicit per-scale
    supervision).
    """

    DEFAULT_SH_DIM = 4
    DEFAULT_HEAD_HIDDEN = 256
    DEFAULT_TAPS = (2, 5, 8, 11)

    def __init__(self, cfg, input_nc: int = 2, pretrained: bool = True,
                 freeze_encoder: bool = False,
                 sh_dim: int = None, head_hidden: int = None,
                 taps=None, **_unused):
        super().__init__(cfg, input_nc=input_nc, pretrained=pretrained,
                         freeze_encoder=freeze_encoder)
        self.sh_dim = int(sh_dim if sh_dim is not None else self.DEFAULT_SH_DIM)
        hidden = int(head_hidden if head_hidden is not None
                     else self.DEFAULT_HEAD_HIDDEN)
        tap_list = tuple(int(t) for t in (taps or self.DEFAULT_TAPS))
        assert all(0 <= t < self.NUM_LAYERS for t in tap_list), \
            f"taps must be in [0, {self.NUM_LAYERS}), got {tap_list}"
        self.taps = tap_list

        self.sh_heads = nn.ModuleList([
            SHHead(feat_dim=2 * self.EMBED_DIM, sh_dim=self.sh_dim,
                   hidden=hidden)
            for _ in tap_list
        ])
        self.sh_mix = nn.Linear(self.sh_dim * len(tap_list), self.sh_dim)

    def forward(self, x, **_unused):
        orig_h, orig_w = x.shape[2], x.shape[3]
        _final, taps = self._encode_with_taps(x, self.taps)

        pred_sh_scales = []
        for idx, head in zip(self.taps, self.sh_heads):
            t = taps[idx]                                  # (B, 1+N, D)
            cls_feat = t[:, 0]
            patch_mean = t[:, 1:].mean(dim=1)
            pred_sh_scales.append(
                head(torch.cat([cls_feat, patch_mean], dim=1)))

        sh_cat = torch.cat(pred_sh_scales, dim=1)          # (B, sh_dim * T)
        pred_sh = self.sh_mix(sh_cat)                      # (B, sh_dim)

        final = self.encoder.ln(taps[self.taps[-1]]) \
            if self.taps[-1] == self.NUM_LAYERS - 1 else _final
        patch_tokens = final[:, 1:]
        pred_depth = self._decode_depth(patch_tokens, orig_h, orig_w)

        return {
            "pred_depth": pred_depth,
            "pred_sh": pred_sh,
            "pred_sh_scales": pred_sh_scales,
        }


class PretrainedViTFOAV6OracleNC3(ViTFOABackbone):
    """v6_oracle_nc3 — ViT oracle with GT energy map concatenated at input (exp219).

    Same as v1 (aux SH head on [cls, mean(patch)]) but ``input_nc=3`` so
    the 1x1 input_adapter maps ``cat(binaural[2ch], energy_map[1ch])`` to
    the 3-channel pseudo-RGB that the pretrained ViT expects. Input
    assembly is done in ``_train_step_oracle``; this model just sees a
    3-channel tensor.
    """

    DEFAULT_SH_DIM = 4
    DEFAULT_HEAD_HIDDEN = 256

    def __init__(self, cfg, input_nc: int = 3, pretrained: bool = True,
                 freeze_encoder: bool = False,
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
        tokens = self._encode(x)
        cls_feat = tokens[:, 0]
        patch_tokens = tokens[:, 1:]
        patch_mean = patch_tokens.mean(dim=1)
        pred_sh = self.sh_head(torch.cat([cls_feat, patch_mean], dim=1))
        pred_depth = self._decode_depth(patch_tokens, orig_h, orig_w)
        return {"pred_depth": pred_depth, "pred_sh": pred_sh}
