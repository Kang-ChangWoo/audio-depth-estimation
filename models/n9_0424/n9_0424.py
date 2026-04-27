"""n9_0424 — Binaural ERP Depth with Implicit Sound-Field Projection Fusion.

Pipeline:

    spectrogram (B, 2, H, W)
      → input_adapter (Conv 2→3, 1×1)      [inside ViTFOABackbone]
      → Spec ViT-B/16 (pretrained, block loop, taps at 4 / 8 / 12)
         → F4, F8, F12 tokens               (each (B, N_patch, D=768))
      → Reshape F12 → (B, D, grid_h, grid_w)
      → ImplicitSoundFieldProjectionFusion  → F12_fused, rep_pred (B, 8, 4)
      → Reshape F12_fused back to tokens
      → RefinementAttention at F12 (Q=fused, K=V=raw)
      → DPT neck (reassemble F4, F8, F12_refined at 4× / 2× / 1×)
      → DepthDecoder → pred_depth (B, 1, H, W)

Forward output dict:
    pred_depth   (B, 1, H, W)
    rep_pred     (B, 8, 4)
    pred_sh      (B, 4)   — first-bin rep_pred exposed for legacy foa-L1
                             (only used if no rep_gt supplied in batch)

Training loss composition (handled by _train_step_foa_0415 extended to
consume rep_gt from the 5-tuple batch when use_distance_bins=True):

    L_depth          = BerHu + SILog
    L_rep            = weighted_rep_loss(rep_pred, rep_gt)
    L_energy_map     = energy_map_loss(rep_pred, rep_gt, basis)  [if λ_emap>0]

    L_total = depth_weight·L_depth + lambda_sh·L_rep + lambda_energy_map·L_energy_map

Registration (already wired by the author in utils/train_utils.py):
    _PVITFOA_AUX_SH_NAMES  ← 'n9_0424'
    _PVITFOA_CLASSES       ← 'n9_0424': N9_0424Net
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.pretrain.pretrained_vit_foa import ViTFOABackbone
from models.renew.renew_single import (
    MultiScaleNeck, DepthDecoder, RefinementAttention,
)
from .fusion import ImplicitSoundFieldProjectionFusion


class N9_0424Net(nn.Module):
    """Spec-ViT encoder + Implicit Sound-Field Projection Fusion + DPT decoder.

    Parameters
    ----------
    cfg : SimpleNamespace-like
        cfg.dataset.images_size   (H, W) of ERP grid — e.g. [256, 512]
        cfg.dataset.depth_norm    whether to sigmoid the depth output
    input_nc : int
        Channel count of audio input. Default 2 (binaural).
    pretrained : bool
        Load ImageNet ViT-B/16 weights.
    freeze_encoder : bool
        Freeze the ViT encoder. Default False.
    taps : tuple[int]
        1-indexed ViT block taps. Default (4, 8, 12) — matches RenewSingle.
    fusion_ch : int
        Channel dim of DPT neck/decoder (default 256).
    attn_heads : int
        Heads in the F12 refinement cross-attention (default 4).
    K : int
        Distance-bin count (default 8). Fusion.DEFAULT_GATE must match.
    D : int
        Implicit per-bin latent dim (default 128).
    res_scale_init : float
        Initial value of the residual-scale parameter in fusion. Default 0.1.
    gate_learnable : bool
        If False, the per-bin gate is fixed at DEFAULT_GATE. Used by
        Variant E (gate ablation).
    enable_fusion : bool
        If False, fusion returns F_in unchanged (rep_pred still emitted).
        Used by Variant A (baseline) and Variant B (aux-only).
    enable_refinement : bool
        If False (default), skip the F12 cross-attention refinement. This
        matters for *clean* ablations: if refinement is always on, a
        "baseline" run has ViT+Attn+DPT rather than ViT+DPT, which mixes
        two architectural factors. Keep this off for the initial A–E
        sweep; turn it on only in follow-up experiments that specifically
        isolate the effect of refinement on a confirmed-good fusion model.
    """

    TAPS_DEFAULT = (4, 8, 12)
    N_LAYERS = 12
    EMBED_DIM = 768

    def __init__(self, cfg, input_nc: int = 2, pretrained: bool = True,
                 freeze_encoder: bool = False,
                 taps=None, fusion_ch: int = 256, attn_heads: int = 4,
                 K: int = 8, D: int = 128,
                 res_scale_init: float = 0.1,
                 gate_learnable: bool = True,
                 enable_fusion: bool = True,
                 enable_refinement: bool = False,
                 **_unused):
        super().__init__()
        taps = tuple(int(t) for t in (taps or self.TAPS_DEFAULT))
        assert all(1 <= t <= self.N_LAYERS for t in taps), \
            f"taps must be in [1, {self.N_LAYERS}], got {taps}"
        self.taps = taps
        self.K = int(K)
        self.enable_fusion = enable_fusion
        self.enable_refinement = enable_refinement

        # (1) Spec ViT — sole encoder.
        self.vit_spec = ViTFOABackbone(
            cfg, input_nc=input_nc, pretrained=pretrained,
            freeze_encoder=freeze_encoder)
        # Backbone's own decoder unused here.
        del self.vit_spec.decoder

        self.grid_h = self.vit_spec.grid_h
        self.grid_w = self.vit_spec.grid_w

        # (2) Fusion block on F12 feature grid.
        self.fusion = ImplicitSoundFieldProjectionFusion(
            C=self.EMBED_DIM, H=self.grid_h, W=self.grid_w,
            K=self.K, D=int(D),
            res_scale_init=float(res_scale_init),
            gate_learnable=bool(gate_learnable),
            enable_fusion=bool(enable_fusion))

        # (3) Refinement cross-attention at F12. Light (1 layer, 4 heads).
        # Built unconditionally so checkpoints can turn it on/off without a
        # shape change; gated at forward time by self.enable_refinement.
        self.refinement = RefinementAttention(
            embed_dim=self.EMBED_DIM, num_heads=int(attn_heads))

        # (4) DPT neck + depth decoder — reused from renew_single.
        self.neck = MultiScaleNeck(
            embed_dim=self.EMBED_DIM, out_ch=int(fusion_ch),
            grid_h=self.grid_h, grid_w=self.grid_w)
        self.decoder = DepthDecoder(
            ch=int(fusion_ch),
            target_h=self.vit_spec.target_h,
            target_w=self.vit_spec.target_w,
            depth_norm=getattr(cfg.dataset, 'depth_norm', True))

    def _run_vit_with_taps(self, backbone: ViTFOABackbone,
                           x: torch.Tensor) -> dict:
        """Walk the ViT encoder; surface per-tap outputs."""
        tokens = backbone._embed(x)                       # (B, 1+N, D)
        tokens = tokens + backbone.encoder.pos_embedding
        tokens = backbone.encoder.dropout(tokens)
        tap_set = set(self.taps)
        taps_out = {}
        for i, block in enumerate(backbone.encoder.layers, start=1):
            tokens = block(tokens)
            if i in tap_set:
                taps_out[i] = tokens
        final = backbone.encoder.ln(tokens)
        if self.N_LAYERS in tap_set:
            taps_out[self.N_LAYERS] = final
        taps_out['final'] = final
        return taps_out

    def forward(self, x: torch.Tensor, **_unused) -> dict:
        orig_h, orig_w = x.shape[2], x.shape[3]

        # 1. Spec ViT taps.
        spec_taps = self._run_vit_with_taps(self.vit_spec, x)
        F4 = spec_taps[self.taps[0]][:, 1:]       # drop CLS
        F8 = spec_taps[self.taps[1]][:, 1:]
        F12 = spec_taps[self.taps[-1]][:, 1:]     # (B, N, D)

        # 2. Fusion at F12 (tokens ↔ spatial).
        B, N, D = F12.shape
        assert N == self.grid_h * self.grid_w, \
            f"patch count mismatch: N={N}, grid={self.grid_h}×{self.grid_w}"
        F12_2d = F12.transpose(1, 2).reshape(B, D, self.grid_h, self.grid_w)
        F12_fused_2d, rep_pred = self.fusion(F12_2d)
        F12_fused = F12_fused_2d.flatten(2).transpose(1, 2)  # back to tokens

        # 3. Optional refinement cross-attention (Q=fused, K=V=raw F12).
        #    Off by default so A–E ablations isolate fusion from attention.
        if self.enable_refinement:
            F12_out = self.refinement(F12_fused, F12)
        else:
            F12_out = F12_fused

        # 4. DPT neck + decoder.
        m4, m8, m12 = self.neck(F4, F8, F12_out)
        pred_depth = self.decoder(m4, m8, m12)
        if pred_depth.shape[-2:] != (orig_h, orig_w):
            pred_depth = F.interpolate(pred_depth, size=(orig_h, orig_w),
                                       mode='bilinear', align_corners=False)

        return {
            'pred_depth': pred_depth,
            'rep_pred':   rep_pred,                      # (B, 8, 4)
            # Exposed so _train_step_foa_0415 legacy path doesn't crash if
            # rep_gt is missing from the batch (e.g. use_distance_bins=False
            # was left unset). When rep_gt is present, the train step
            # prefers the rep_pred-based loss and this is unused.
            'pred_sh':    rep_pred[:, 0, :].contiguous(),   # (B, 4)
        }


if __name__ == '__main__':
    from types import SimpleNamespace as NS
    cfg = NS(dataset=NS(images_size=[256, 512], depth_norm=True))
    net = N9_0424Net(cfg, input_nc=2, pretrained=False).eval()
    x = torch.randn(1, 2, 256, 512)
    with torch.no_grad():
        out = net(x)
        for k, v in out.items():
            print(f'{k:12s}: {tuple(v.shape)}')
