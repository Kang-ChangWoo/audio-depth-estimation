"""N1 (2026-04-20) — ViT × N2-temporal hybrids.

Ports the N2 temporal-FOA mechanisms (early/mid/late energy maps,
per-bin RMS, multi-scale SH, energy-attention supervision) into the
pretrained ViT-B/16 backbone. Four classes:

  PVitN1TemapInput     — 5ch input (binaural + 3 energy maps) -> aux SH
                         head. ViT analogue of n2_temap_input (exp201).
                         Output: {pred_depth, pred_sh}.

  PVitN1TemapRMSFiLM   — 2ch binaural input, 12-dim temporal_rms
                         modulates ViT tokens via FiLM injected between
                         early and late blocks. ViT analogue of
                         n2_temporal_rms_film (exp199/200).
                         Output: {pred_depth, pred_sh}.

  PVitN1TemapEAttn     — 5ch input + three energy heads predicting the
                         per-bin ERP energy maps for explicit
                         supervision (picked up by _train_step_n2 when
                         pred_temporal_energies is in the output).
                         ViT analogue of n2_temporal_energy (exp192).
                         Output: {pred_depth, pred_sh, pred_temporal_energies}.

  PVitN1TemapMSSH      — 5ch input + multi-scale SH heads tapped at
                         four ViT blocks. Combines exp217's MSSH with
                         exp201's temporal-input concatenation.
                         Output: {pred_depth, pred_sh, pred_sh_scales}.

All four route through ``_train_step_n2`` (dataset_n2 7-tuple). The SH
target for every variant is ``temporal_rms`` (12-dim) — unified so the
four differ only in architecture, not in loss target.

Registration:
  - ``_N2_CLASSES`` in utils/train_utils.py picks up each name.
  - train.py ``_train_step_n2`` / ``_val_metrics`` and test.py each add
    the new names to the existing temap_input / rms_film branches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.pretrain.pretrained_vit_foa import ViTFOABackbone, SHHead
from models.pretrain.pretrained_vit_foa_v6 import ViTEnergyHead


class PVitN1TemapInput(ViTFOABackbone):
    """ViT with 5-channel input (binaural + 3 temporal energy maps).

    Simplest temporal variant — the model can choose to ignore the three
    extra channels if they don't help. Same architecture as v1 ViT,
    sh_dim=12 to match temporal_rms.
    """

    DEFAULT_SH_DIM = 12
    DEFAULT_HEAD_HIDDEN = 256

    def __init__(self, cfg, input_nc: int = 5, pretrained: bool = True,
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


class PVitN1TemapRMSFiLM(ViTFOABackbone):
    """ViT with binaural input + FiLM modulation from temporal_rms.

    2ch input (binaural). A small MLP projects the 12-dim temporal_rms
    vector to per-channel (gamma, beta) for FiLM, which modulates the
    ViT tokens between blocks[n_early-1] and blocks[n_early]. Late ViT
    blocks and the depth decoder therefore see features explicitly
    shaped by per-bin energy magnitudes.

    Forward signature matches n2_temporal_rms_film: model(x, trms).
    """

    DEFAULT_SH_DIM = 12
    DEFAULT_HEAD_HIDDEN = 256
    DEFAULT_N_EARLY = 4
    TRMS_DIM = 12

    def __init__(self, cfg, input_nc: int = 2, pretrained: bool = True,
                 freeze_encoder: bool = False,
                 sh_dim: int = None, head_hidden: int = None,
                 n_early: int = None, **_unused):
        super().__init__(cfg, input_nc=input_nc, pretrained=pretrained,
                         freeze_encoder=freeze_encoder)
        self.sh_dim = int(sh_dim if sh_dim is not None else self.DEFAULT_SH_DIM)
        hidden = int(head_hidden if head_hidden is not None
                     else self.DEFAULT_HEAD_HIDDEN)
        self.n_early = int(n_early if n_early is not None
                           else self.DEFAULT_N_EARLY)
        assert 1 <= self.n_early < self.NUM_LAYERS

        self.sh_head = SHHead(feat_dim=2 * self.EMBED_DIM, sh_dim=self.sh_dim,
                              hidden=hidden)
        self.film = nn.Sequential(
            nn.LayerNorm(self.TRMS_DIM),
            nn.Linear(self.TRMS_DIM, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * self.EMBED_DIM),
        )

    def forward(self, x, trms=None, **_unused):
        orig_h, orig_w = x.shape[2], x.shape[3]
        tokens = self._embed(x)
        tokens = tokens + self.encoder.pos_embedding
        tokens = self.encoder.dropout(tokens)

        for i in range(self.n_early):
            tokens = self.encoder.layers[i](tokens)

        if trms is not None:
            gb = self.film(trms)                           # (B, 2D)
            gamma, beta = gb.chunk(2, dim=-1)
            tokens = tokens * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

        for i in range(self.n_early, self.NUM_LAYERS):
            tokens = self.encoder.layers[i](tokens)
        tokens = self.encoder.ln(tokens)

        cls_feat = tokens[:, 0]
        patch_tokens = tokens[:, 1:]
        patch_mean = patch_tokens.mean(dim=1)
        pred_sh = self.sh_head(torch.cat([cls_feat, patch_mean], dim=1))
        pred_depth = self._decode_depth(patch_tokens, orig_h, orig_w)
        return {"pred_depth": pred_depth, "pred_sh": pred_sh}


class PVitN1TemapEAttn(ViTFOABackbone):
    """ViT with 5ch input + three energy heads predicting per-bin ERP maps.

    Same concat input as PVitN1TemapInput, but also emits three
    predicted energy maps (one per temporal bin). The training loop
    supervises them against the GT temporal_energies ((B, 3, H, W))
    when ``pred_temporal_energies`` is in the output dict — the existing
    n2_temporal_energy supervision branch in train.py is extended to
    include this model's name.
    """

    DEFAULT_SH_DIM = 12
    DEFAULT_HEAD_HIDDEN = 256
    N_BINS = 3

    def __init__(self, cfg, input_nc: int = 5, pretrained: bool = True,
                 freeze_encoder: bool = False,
                 sh_dim: int = None, head_hidden: int = None, **_unused):
        super().__init__(cfg, input_nc=input_nc, pretrained=pretrained,
                         freeze_encoder=freeze_encoder)
        self.sh_dim = int(sh_dim if sh_dim is not None else self.DEFAULT_SH_DIM)
        hidden = int(head_hidden if head_hidden is not None
                     else self.DEFAULT_HEAD_HIDDEN)
        self.sh_head = SHHead(feat_dim=2 * self.EMBED_DIM, sh_dim=self.sh_dim,
                              hidden=hidden)
        self.energy_heads = nn.ModuleList([
            ViTEnergyHead(self.EMBED_DIM, self.grid_h, self.grid_w)
            for _ in range(self.N_BINS)
        ])

    def forward(self, x, **_unused):
        orig_h, orig_w = x.shape[2], x.shape[3]
        tokens = self._encode(x)
        cls_feat = tokens[:, 0]
        patch_tokens = tokens[:, 1:]
        patch_mean = patch_tokens.mean(dim=1)
        pred_sh = self.sh_head(torch.cat([cls_feat, patch_mean], dim=1))
        pred_depth = self._decode_depth(patch_tokens, orig_h, orig_w)

        bin_maps = [head(patch_tokens, orig_h, orig_w)
                    for head in self.energy_heads]
        pred_temporal_energies = torch.cat(bin_maps, dim=1)

        return {
            "pred_depth": pred_depth,
            "pred_sh": pred_sh,
            "pred_temporal_energies": pred_temporal_energies,
        }


class PVitN1TemapMSSH(ViTFOABackbone):
    """ViT with 5ch input + multi-scale SH heads at 4 tap layers.

    Combines PVitN1TemapInput's 5ch concat with v6_mssh's multi-scale SH
    supervision (taps [2, 5, 8, 11] by default, one SHHead per tap,
    learnable linear mix -> fused ``pred_sh``). Tests whether multi-scale
    directional supervision stacks with temporal-input enrichment.
    """

    DEFAULT_SH_DIM = 12
    DEFAULT_HEAD_HIDDEN = 256
    DEFAULT_TAPS = (2, 5, 8, 11)

    def __init__(self, cfg, input_nc: int = 5, pretrained: bool = True,
                 freeze_encoder: bool = False,
                 sh_dim: int = None, head_hidden: int = None,
                 taps=None, **_unused):
        super().__init__(cfg, input_nc=input_nc, pretrained=pretrained,
                         freeze_encoder=freeze_encoder)
        self.sh_dim = int(sh_dim if sh_dim is not None else self.DEFAULT_SH_DIM)
        hidden = int(head_hidden if head_hidden is not None
                     else self.DEFAULT_HEAD_HIDDEN)
        tap_list = tuple(int(t) for t in (taps or self.DEFAULT_TAPS))
        assert all(0 <= t < self.NUM_LAYERS for t in tap_list)
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
            t = taps[idx]
            cls_feat = t[:, 0]
            patch_mean = t[:, 1:].mean(dim=1)
            pred_sh_scales.append(
                head(torch.cat([cls_feat, patch_mean], dim=1)))

        sh_cat = torch.cat(pred_sh_scales, dim=1)
        pred_sh = self.sh_mix(sh_cat)

        final = self.encoder.ln(taps[self.taps[-1]]) \
            if self.taps[-1] == self.NUM_LAYERS - 1 else _final
        patch_tokens = final[:, 1:]
        pred_depth = self._decode_depth(patch_tokens, orig_h, orig_w)

        return {
            "pred_depth": pred_depth,
            "pred_sh": pred_sh,
            "pred_sh_scales": pred_sh_scales,
        }
