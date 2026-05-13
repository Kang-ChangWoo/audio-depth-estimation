"""Renew-single model — causal dual-ViT with explicit sound-field bottleneck.

Pipeline (everything is in the forward path of depth loss)
----------------------------------------------------------

    spectrogram (B, 2, H, W)
      → input_adapter_spec (Conv 2→3, 1×1)
      → Spec ViT-B/16 (pretrained, block loop, taps at layers 4 / 8 / 12)
         → F4_spec, F8_spec, F12_spec     (B, N_patch, D=768)
         → final_spec (post-LN)            (B, 1+N_patch, D)

      → FOAHead(CLS_final ⊕ mean(patch_final))
         → k_n  (B, sh_dim)                 sh_dim = (order+1)²  (default 36, order 5)

      → SoundFieldGenerator(k_n)  → Ŝ       (B, 1, H, W)
      → (optional teacher-forcing mix)
             S_in = α·S_gt + (1-α)·Ŝ       α = cfg.model.teacher_ratio (or kwarg)

      → input_adapter_sf (Conv 1→3, 1×1)
      → Sound-field ViT-B/16 (pretrained, block loop, taps at 4/8/12)
         → F4_sf,  F8_sf,  F12_sf          (B, N_patch, D)

    Per-scale fusion (FusionBlock, additive-residual):
         F*_fused = LN(F*_spec + proj_sf(F*_sf))

    Refinement cross-attention at bottleneck:
         F12_fused <- RefinementAttention(Q=F12_fused, K=V=F12_sf)

    DPT neck  → reassemble F4/F8/F12 to spatial maps at 4× / 2× / 1× grid
    Depth decoder  → top-down RefineNet fuse + progressive upsample
    Depth head  → pred_depth (B, 1, H, W)

Output dict
-----------
    pred_depth    (B, 1, H, W)
    pred_sh       (B, sh_dim)    alias for k_n; foa_0415 train step L1's
                                 the first 4 dims against gt_foa
    pred_energy   (B, 1, H, W)   Ŝ BEFORE teacher mixing — the raw
                                 generated sound field, L1'd against
                                 gt_energy_map when lambda_energy > 0
    k_n           (B, sh_dim)    explicit name
    sf_in         (B, 1, H, W)   what the Sound-field ViT actually saw
                                 (== pred_energy when teacher_ratio=0)

Causality and teacher forcing
-----------------------------
This network is end-to-end: depth gradient must flow through the
sound-field generator back to the FOA head and spectrogram backbone.
The teacher-forcing mixing lets you blend GT energy map into the SF ViT
input *as a curriculum* (α = 1 → GT only early, α = 0 → predicted only
late). When teacher_ratio > 0, the depth-loss path through S_in is
partially short-circuited to S_gt (no gradient through Ŝ for that
fraction), which is intentional for early-epoch stability.

Routing
-------
Register `renew_single` in utils/train_utils.py:
  _PVITFOA_AUX_SH_NAMES   → goes through _train_step_foa_0415 / is_foa_0415_model
  _PVITFOA_CLASSES        → built with pretrained/freeze_encoder ViT kwargs
The foa_0415 train step already handles:
    L = depth_weight · L_depth(BerHu+SILog)
      + lambda_sh    · L1(pred_sh[:, :4], gt_foa)
      + lambda_energy · L1(pred_energy, gt_energy_map)   (if λ > 0)
If you want a distribution-KL term on depth histograms (L_kl) and
teacher-forcing α drawn from the training epoch, that's added by a
tiny extension in train.py — kept out of this file to stay honest about
"do not change the main data flow" unless you opt in.

ViT modification
----------------
No wrappers. We instantiate two `ViTFOABackbone`s (the existing shared
backbone under `models/pretrain/pretrained_vit_foa.py`) and walk their
`self.encoder.layers` list directly in a block loop. The backbone
already exposes `_embed`, `encoder.pos_embedding`, `encoder.dropout`,
and `encoder.ln` — the same pieces `torchvision.models.vit.Encoder.forward`
uses internally, so pretrained weights behave identically. The
`decoder` attribute the backbone allocates is unused here and is
deleted to save memory.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.pretrain.pretrained_vit_foa import ViTFOABackbone
from data.sh_basis import sh_basis_matrix


# ============================================================
# Heads / generators / fusion
# ============================================================

class FOAHead(nn.Module):
    """[CLS ⊕ mean(patch)] → LN → Linear → GELU → Linear → k_n."""

    def __init__(self, embed_dim: int = 768, sh_dim: int = 36,
                 hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(2 * embed_dim),
            nn.Linear(2 * embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, sh_dim),
        )

    def forward(self, cls_feat: torch.Tensor, patch_feat: torch.Tensor) -> torch.Tensor:
        patch_mean = patch_feat.mean(dim=1)                # (B, D)
        x = torch.cat([cls_feat, patch_mean], dim=1)       # (B, 2D)
        return self.net(x)                                 # (B, sh_dim)


class SoundFieldGenerator(nn.Module):
    """k_n → approximated sound field on ERP grid via a fixed SH basis.

    Two reconstruction modes:
      'rank1' (default) — E(Ω) = (y(Ω)ᵀ · k_n)² , peaked.
      'diag'            — E(Ω) = Σ kᵢ² · yᵢ(Ω)² , diffuse.

    The SH basis B = sh_basis_matrix(order, el, az) is (H·W, sh_dim)
    and registered as a non-trainable buffer so it travels on the same
    device as the model and is stored in ``state_dict``.
    """

    def __init__(self, sh_dim: int = 36, H: int = 256, W: int = 512,
                 mode: str = 'rank1', normalize: bool = True):
        super().__init__()
        order = int(round(math.sqrt(sh_dim))) - 1
        assert (order + 1) ** 2 == sh_dim, \
            f"sh_dim must be (order+1)^2; got sh_dim={sh_dim}"
        el = np.linspace(np.pi / 2, -np.pi / 2, H)
        az = np.linspace(-np.pi, np.pi, W)
        az_grid, el_grid = np.meshgrid(az, el)
        basis = sh_basis_matrix(order, el_grid, az_grid)    # (H*W, sh_dim)
        self.register_buffer('basis', torch.from_numpy(basis.astype(np.float32)))
        self.H, self.W = H, W
        self.mode = mode
        self.normalize = normalize
        self.order = order

    def forward(self, k_n: torch.Tensor) -> torch.Tensor:
        """k_n: (B, sh_dim) → sf: (B, 1, H, W)"""
        if self.mode == 'diag':
            basis_sq = self.basis ** 2
            e = (k_n ** 2) @ basis_sq.T                    # (B, H*W)
        else:
            proj = k_n @ self.basis.T                      # (B, H*W)
            e = proj ** 2
        sf = e.view(k_n.shape[0], 1, self.H, self.W)
        if self.normalize:
            peaks = sf.flatten(1).abs().max(dim=1).values.clamp_min(1e-8)
            sf = sf / peaks.view(-1, 1, 1, 1)
        return sf


class FusionBlock(nn.Module):
    """Additive-residual fusion of spec and sound-field patch tokens.

    f_fused = LayerNorm(f_spec + proj_sf(f_sf))

    The user's design note: `concat + linear` was the original ask but
    additive-residual is closer to the diagram (sound-field branch feeds
    *into* the main spec stream) and keeps param count lower.
    """

    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.proj_sf = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, f_spec: torch.Tensor, f_sf: torch.Tensor) -> torch.Tensor:
        return self.norm(f_spec + self.proj_sf(f_sf))


class RefinementAttention(nn.Module):
    """Cross-attention at the F12 bottleneck.  Q=fused, K=V=f_sf.

    Intentionally small (1 layer, 4 heads) — the user's earlier caveat
    "no heavy cross-attention" still applies; this is one lightweight
    refinement step that matches the ``Attn`` block in the diagram.
    """

    def __init__(self, embed_dim: int = 768, num_heads: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(q, kv, kv, need_weights=False)
        return self.norm(q + out)


class ReassembleBlock(nn.Module):
    """DPT reassemble: tokens → 2D map at a target spatial resolution.

    Project D → out_ch (1×1 conv), then spatially ×scale via a stack of
    ConvTranspose 4/2/1 steps (one per ×2).
    """

    def __init__(self, embed_dim: int, out_ch: int, grid_h: int, grid_w: int,
                 scale: int):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.project = nn.Conv2d(embed_dim, out_ch, 1)
        if scale == 1:
            self.resize = nn.Identity()
        else:
            layers, ch, s = [], out_ch, scale
            while s > 1:
                layers.append(nn.ConvTranspose2d(ch, ch, 4, 2, 1))
                s //= 2
            self.resize = nn.Sequential(*layers)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, N, D = tokens.shape
        x = tokens.transpose(1, 2).reshape(B, D, self.grid_h, self.grid_w)
        x = self.project(x)
        return self.resize(x)


class FeatureFusionBlock(nn.Module):
    """RefineNet residual unit with top-down + sum fusion."""

    def __init__(self, ch: int):
        super().__init__()
        self.res = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, deep: torch.Tensor, shallow: torch.Tensor) -> torch.Tensor:
        if deep.shape[-2:] != shallow.shape[-2:]:
            deep = F.interpolate(deep, size=shallow.shape[-2:],
                                 mode='bilinear', align_corners=False)
        x = deep + shallow
        return F.relu(x + self.res(x), inplace=True)


class MultiScaleNeck(nn.Module):
    def __init__(self, embed_dim: int, out_ch: int, grid_h: int, grid_w: int):
        super().__init__()
        self.r4  = ReassembleBlock(embed_dim, out_ch, grid_h, grid_w, scale=4)
        self.r8  = ReassembleBlock(embed_dim, out_ch, grid_h, grid_w, scale=2)
        self.r12 = ReassembleBlock(embed_dim, out_ch, grid_h, grid_w, scale=1)

    def forward(self, F4, F8, F12):
        return self.r4(F4), self.r8(F8), self.r12(F12)


class DepthDecoder(nn.Module):
    def __init__(self, ch: int, target_h: int, target_w: int,
                 depth_norm: bool = True):
        super().__init__()
        self.target_h = target_h
        self.target_w = target_w
        self.depth_norm = depth_norm
        self.fuse_12_8 = FeatureFusionBlock(ch)
        self.fuse_8_4  = FeatureFusionBlock(ch)
        self.post = nn.Sequential(
            nn.ConvTranspose2d(ch, ch // 2, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(ch // 2, ch // 4, 4, 2, 1),
            nn.ReLU(True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(ch // 4, ch // 8, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(ch // 8, 1, 1),
        )

    def forward(self, m4, m8, m12):
        x = self.fuse_12_8(m12, m8)          # (B, ch, 2g_h, 2g_w)
        x = self.fuse_8_4(x,  m4)            # (B, ch, 4g_h, 4g_w)
        x = self.post(x)                     # (B, ch/4, 16g_h, 16g_w)
        x = self.head(x)                     # (B, 1,  16g_h, 16g_w)
        if self.depth_norm:
            x = torch.sigmoid(x)
        if x.shape[-2:] != (self.target_h, self.target_w):
            x = F.interpolate(x, size=(self.target_h, self.target_w),
                              mode='bilinear', align_corners=False)
        return x


# ============================================================
# Full model
# ============================================================

class RenewSingleNet(nn.Module):
    """See module docstring for the full architecture rationale.

    Parameters
    ----------
    cfg : SimpleNamespace-like
        cfg.dataset.images_size      (H, W) of ERP grid (e.g. [256, 512])
        cfg.dataset.depth_norm       whether to sigmoid the depth output
    input_nc : int
        Channel count of the audio input (default 2 for binaural stereo).
    pretrained : bool
        Load ImageNet ViT-B/16 weights for both ViTs.
    freeze_encoder : bool
        Freeze both ViT encoders.
    sh_dim : int
        Coefficient count. Default 36 → order-5 SH. Must be (order+1)².
    head_hidden : int
        FOA-head hidden dim.
    taps : tuple[int]
        1-indexed ViT block indices to tap. Default (4, 8, 12).
    sf_mode : str
        'rank1' (default, peaked) or 'diag' (diffuse).
    teacher_ratio : float
        Default α. Value in forward kwargs overrides this. Between 0–1.
        α=0 → always use predicted sf. α=1 → always GT (requires gt_energy).
    fusion_ch : int
        Channel dim of the DPT reassemble/fusion stack (default 256).
    attn_heads : int
        Heads in the refinement cross-attention at F12.
    """

    TAPS_DEFAULT = (4, 8, 12)
    N_LAYERS = 12
    EMBED_DIM = 768

    def __init__(self, cfg, input_nc: int = 2, pretrained: bool = True,
                 freeze_encoder: bool = False,
                 sh_dim: int = 36, head_hidden: int = 512,
                 taps=None, sf_mode: str = 'rank1',
                 teacher_ratio: float = 0.0,
                 fusion_ch: int = 256, attn_heads: int = 4,
                 disable_fusion: bool = False,
                 **_unused):
        super().__init__()
        self.disable_fusion = disable_fusion
        taps = tuple(int(t) for t in (taps or self.TAPS_DEFAULT))
        assert all(1 <= t <= self.N_LAYERS for t in taps), \
            f"taps must be in [1, {self.N_LAYERS}], got {taps}"
        self.taps = taps
        self.default_teacher_ratio = float(teacher_ratio)

        # (1) Spec ViT and (4) Sound-field ViT — both instances of the
        # shared ``ViTFOABackbone``. No wrapping; we iterate their
        # encoder.layers directly in _run_vit_with_taps.
        self.vit_spec = ViTFOABackbone(
            cfg, input_nc=input_nc, pretrained=pretrained,
            freeze_encoder=freeze_encoder)
        self.vit_sf = ViTFOABackbone(
            cfg, input_nc=1, pretrained=pretrained,
            freeze_encoder=freeze_encoder)
        # Each backbone builds its own decoder; unused here.
        del self.vit_spec.decoder
        del self.vit_sf.decoder

        self.grid_h = self.vit_spec.grid_h
        self.grid_w = self.vit_spec.grid_w

        # (2) FOA head → k_n (sh_dim)
        self.foa_head = FOAHead(embed_dim=self.EMBED_DIM, sh_dim=sh_dim,
                                hidden=head_hidden)

        # (3) Sound-field generator (fixed SH basis buffer)
        self.sf_gen = SoundFieldGenerator(
            sh_dim=sh_dim, H=self.vit_spec.target_h, W=self.vit_spec.target_w,
            mode=sf_mode, normalize=True)

        # (5) Per-scale additive-residual fusion blocks (one per tap)
        self.fusions = nn.ModuleList([
            FusionBlock(embed_dim=self.EMBED_DIM) for _ in taps
        ])

        # (6) Refinement cross-attention at the F12 bottleneck
        self.refinement = RefinementAttention(embed_dim=self.EMBED_DIM,
                                              num_heads=attn_heads)

        # (7) DPT multi-scale neck + depth decoder
        self.neck = MultiScaleNeck(embed_dim=self.EMBED_DIM, out_ch=fusion_ch,
                                   grid_h=self.grid_h, grid_w=self.grid_w)
        self.decoder = DepthDecoder(
            ch=fusion_ch,
            target_h=self.vit_spec.target_h,
            target_w=self.vit_spec.target_w,
            depth_norm=getattr(cfg.dataset, 'depth_norm', True))

        self.sh_dim = int(sh_dim)

    # ---- ViT block loop with taps (same pattern as torchvision's
    # Encoder.forward, but surfaces per-block outputs) ----
    def _run_vit_with_taps(self, backbone: ViTFOABackbone,
                           x: torch.Tensor) -> dict:
        tokens = backbone._embed(x)                           # (B, 1+N, D)
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
            taps_out[self.N_LAYERS] = final        # post-LN for the last tap
        taps_out['final'] = final
        return taps_out

    def forward(self, x: torch.Tensor,
                gt_energy: torch.Tensor = None,
                teacher_ratio: float = None,
                **_unused) -> dict:
        """
        x             (B, 2, H, W)     binaural spectrogram
        gt_energy     (B, 1, H, W)     optional — used when α > 0
        teacher_ratio float            override for the cfg default; in [0, 1]
        """
        orig_h, orig_w = x.shape[2], x.shape[3]
        alpha = self.default_teacher_ratio if teacher_ratio is None \
            else float(teacher_ratio)
        if alpha > 0 and gt_energy is None:
            alpha = 0.0              # no GT available → fall back to pred

        # ---- Upper branch: spectrogram → k_n → Ŝ ----
        spec_taps = self._run_vit_with_taps(self.vit_spec, x)
        final_spec = spec_taps['final']
        k_n = self.foa_head(final_spec[:, 0], final_spec[:, 1:])   # (B, sh_dim)

        pred_sf = self.sf_gen(k_n)                                  # (B, 1, H, W)

        # Teacher-forcing mix: depth gradient bypasses Ŝ for the α
        # fraction, keeping the causal path alive for the (1-α)
        # fraction. α is a knob the training loop can schedule across
        # epochs; at α=0 the whole graph stays end-to-end differentiable.
        if alpha > 0:
            sf_in = alpha * gt_energy + (1.0 - alpha) * pred_sf
        else:
            sf_in = pred_sf

        # ---- Lower branch: sound field → F4/F8/F12 ----
        sf_taps = self._run_vit_with_taps(self.vit_sf, sf_in)

        # ---- Per-scale additive fusion ----
        fused = []
        for tap, fuse in zip(self.taps, self.fusions):
            f_spec = spec_taps[tap][:, 1:]          # drop CLS → (B, N, D)
            f_sf   = sf_taps[tap][:,   1:]
            fused.append(fuse(f_spec, f_sf))        # (B, N, D)

        # ---- Refinement cross-attention at the deepest tap ----
        F4, F8, F12 = fused
        F12 = self.refinement(F12, sf_taps[self.taps[-1]][:, 1:])

        # ---- DPT neck (tokens → 2D maps) → depth decoder ----
        m4, m8, m12 = self.neck(F4, F8, F12)
        pred_depth = self.decoder(m4, m8, m12)
        if pred_depth.shape[-2:] != (orig_h, orig_w):
            pred_depth = F.interpolate(pred_depth, size=(orig_h, orig_w),
                                       mode='bilinear', align_corners=False)

        return {
            'pred_depth':  pred_depth,
            'pred_sh':     k_n,          # first 4 dims used by foa_0415 L_sh
            'pred_energy': pred_sf,      # supervised against gt_energy_map
            'k_n':         k_n,          # same tensor, explicit alias
            'sf_in':       sf_in,        # what the sf-ViT actually received
        }


# ============================================================
# Dummy forward — run directly to sanity-check shapes
# ============================================================
if __name__ == '__main__':
    from types import SimpleNamespace as NS
    cfg = NS(dataset=NS(images_size=[256, 512], depth_norm=True))
    net = RenewSingleNet(cfg, input_nc=2, pretrained=False,
                         sh_dim=36, sf_mode='rank1',
                         teacher_ratio=0.0).eval()
    x    = torch.randn(1, 2, 256, 512)
    gt_e = torch.rand(1, 1, 256, 512)
    with torch.no_grad():
        out = net(x)
        print('pred_depth  :', out['pred_depth'].shape)
        print('pred_sh     :', out['pred_sh'].shape)       # (1, 36)
        print('pred_energy :', out['pred_energy'].shape)

        # Teacher-forcing path (α=0.7 mix)
        out2 = net(x, gt_energy=gt_e, teacher_ratio=0.7)
        print('α=0.7 sf_in :', out2['sf_in'].shape)
