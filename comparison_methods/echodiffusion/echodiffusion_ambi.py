"""EchoDiffusionAmbi — EchoDiffusion backbone with bin-gated FOA conditioning.

Same diffusion-UNet feature-extractor backbone as EchoDiffusion (Zhang et al.),
but the CIDE waveform-conditioning branch is replaced by per-bin gated oracle
FOA representations — the same conditioning signal that n4_0425 uses.

Two ambisonic injection modes (cfg.model.foa_mode):

  'input'      — Build a 1-channel gated energy map from rep_gt the same way
                 n4_0425 / n9_0425 do, concat with the binaural spec → 3-channel
                 ASPP+ASFF input. Cross-attention context is a learnable null
                 token (B, 1, 768) so the diffusion-UNet's transformer blocks
                 still get a context input but no FOA tokens.

  'condition'  — Binaural spec stays 2-channel. Per-bin gated rep_gt (B, K, 4)
                 is projected by a small MLP into K cross-attention tokens of
                 width 768 → context = (B, K, 768). The diffusion-UNet's
                 transformer blocks attend over the K bin tokens. No null token.

Bin gate semantics are identical to n4_0425 (learnable K-vector or fixed
gate_mask buffer; sparsity loss `lambda_sparsity * sigmoid(gate).mean()` is
applied by `_train_step_foa_0415` because the forward returns 'gate').

Forward returns dict (matches n4_0425 contract → routes via
_train_step_foa_0415):
    pred_depth   (B, 1, H, W)
    gate         (K,)            sigmoid(gate_logit)
    rep_pred     (B, K, 4)       a copy of rep_gt (this is an ORACLE model,
                                 same as n4_0425) — kept so train step can
                                 evaluate weighted_rep_loss(rep_pred, rep_gt)
                                 (which is identically zero here) and so test
                                 utils can compute FOA error metrics.
    pred_sh      (B, 4)          rep_pred[:, 0, :], legacy compat stub.

Construction kwargs (passed via cfg.model.* + CLI overrides):
    foa_mode        : str         'input' or 'condition'.
    K               : int         distance bins (must match dataset rep_K).
    gate_init       : float       initial gate_logit value.
    gate_mask       : list[float] | None  fixed per-bin mask, frozen.
    max_depth       : float       output rescaling (matches EchoDiffusion).
    embed_dim       : int         decoder width (default 192, same as parent).
    emb_dim         : int         cross-attention context_dim (default 768).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aspp_asff import UNetASPPASFF
from .diffusion_unet import DiffusionUNet
from .echodiffusion import Decoder, CIDE


def make_foa_basis_erp(H: int, W: int,
                       device=None, dtype=torch.float32) -> torch.Tensor:
    """FOA basis on an ERP grid, channel order [W, Y, Z, X]. Shape (4, H, W).

        basis[0] = c0                      (W, isotropic)
        basis[1] = c1 · cos(el) · sin(az)  (Y)
        basis[2] = c1 · sin(el)            (Z)
        basis[3] = c1 · cos(el) · cos(az)  (X)

    c0 = 1/√(4π),  c1 = √(3/(4π)).

    Inlined from former models.n9_0424.fusion when n9_0424 was moved to
    models/deprecated/ (Phase E1). Keep in sync if FOA conventions change.
    """
    ys = torch.linspace(0.5, H - 0.5, H, device=device, dtype=dtype) / H
    xs = torch.linspace(0.5, W - 0.5, W, device=device, dtype=dtype) / W

    az = (xs - 0.5) * 2.0 * torch.pi
    el = (0.5 - ys) * torch.pi

    el_grid, az_grid = torch.meshgrid(el, az, indexing='ij')

    x = torch.cos(el_grid) * torch.cos(az_grid)
    y = torch.cos(el_grid) * torch.sin(az_grid)
    z = torch.sin(el_grid)

    c0 = 0.28209479177387814    # 1 / √(4π)
    c1 = 0.4886025119029199     # √(3 / (4π))

    basis = torch.stack([
        torch.full_like(x, c0),
        c1 * y,
        c1 * z,
        c1 * x,
    ], dim=0)
    return basis


def _build_gated_em(rep: torch.Tensor, g: torch.Tensor,
                    foa_basis: torch.Tensor) -> torch.Tensor:
    """rep (B,K,4) × g (K,) → per-bin energy map (B,1,H,W), peak-normed."""
    rep_gated = rep * g.view(1, -1, 1)
    signed = torch.einsum('bkc,chw->bkhw', rep_gated, foa_basis)
    energy_per_bin = torch.log1p(signed.square())
    em = energy_per_bin.sum(dim=1, keepdim=True)
    emax = em.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    return em / emax


class _BinTokenProjector(nn.Module):
    """rep (B, K, 4) × g (K,) → K cross-attention tokens (B, K, emb_dim).

    Each bin gets its own MLP; the K projections share dimensionality but not
    weights, so each bin token can develop its own meaning to the transformer.
    Equivalent to a per-bin Linear(4, emb_dim) implemented as a single Conv1d
    with K groups.
    """

    def __init__(self, K=8, emb_dim=768, hidden=128):
        super().__init__()
        self.K = K
        # Implement K independent (4 → hidden → emb_dim) MLPs as grouped conv1d
        # over the K-bin axis. Input is reshaped to (B, K*4, 1).
        self.fc1 = nn.Conv1d(K * 4,        K * hidden, kernel_size=1, groups=K)
        self.act = nn.GELU()
        self.fc2 = nn.Conv1d(K * hidden,   K * emb_dim, kernel_size=1, groups=K)
        self.hidden = hidden
        self.emb_dim = emb_dim

    def forward(self, rep: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        # rep: (B, K, 4), g: (K,)
        B = rep.shape[0]
        rep_gated = rep * g.view(1, -1, 1)              # (B, K, 4)
        x = rep_gated.reshape(B, self.K * 4, 1)         # (B, K*4, 1)
        x = self.act(self.fc1(x))                       # (B, K*hidden, 1)
        x = self.fc2(x)                                 # (B, K*emb_dim, 1)
        x = x.reshape(B, self.K, self.emb_dim)          # (B, K, emb_dim)
        return x


class EcoDepthEncoderAmbi(nn.Module):
    """ASPP+ASFF spectrogram encoder + diffusion UNet feature extractor with
    bin-gated FOA conditioning. Drops the CIDE/Wav2Vec2 branch.
    """

    def __init__(self, *, K=8, foa_mode='condition',
                 spec_in_channels=2, out_dim=1536,
                 ldm_prior=(32, 64, 256), emb_dim=768, use_cide=False):
        super().__init__()
        assert foa_mode in ('input', 'condition'), \
            f"foa_mode must be 'input' or 'condition', got {foa_mode!r}"
        self.foa_mode = foa_mode
        self.K = K
        self.use_cide = bool(use_cide)

        self.layer1 = nn.Sequential(
            nn.Conv2d(ldm_prior[0], ldm_prior[0], 3, stride=2, padding=1),
            nn.GroupNorm(16, ldm_prior[0]),
            nn.ReLU(),
            nn.Conv2d(ldm_prior[0], ldm_prior[0], 3, stride=2, padding=1),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(ldm_prior[1], ldm_prior[1], 3, stride=2, padding=1),
        )
        self.out_layer = nn.Sequential(
            nn.Conv2d(sum(ldm_prior), out_dim, 1),
            nn.GroupNorm(16, out_dim),
            nn.ReLU(),
        )

        # Spectrogram encoder. spec_in_channels is 2 ('condition' mode) or 3
        # ('input' mode: binaural + gated_em concatenated).
        self.aspp_asff = UNetASPPASFF(in_channels=spec_in_channels, base_c=64)
        self.latent_proj = nn.Sequential(
            nn.Conv2d(128, 512, 1),
            nn.GroupNorm(32, 512),
            nn.SiLU(),
        )

        # Conditioning source.
        if foa_mode == 'condition':
            self.bin_proj = _BinTokenProjector(K=K, emb_dim=emb_dim, hidden=128)
            self.null_context = None
        else:  # 'input'
            # No bin token projection; FOA enters via the spec input.
            self.null_context = nn.Parameter(torch.randn(1, 1, emb_dim) * 0.02)
            self.bin_proj = None
        # Optional CIDE/Wav2Vec2 branch — when use_cide=True the binaural
        # waveform is projected to a single (B, 1, emb_dim) token that is
        # PREPENDED to the cross-attention context (additive on top of FOA).
        # When False, the original null/bin-only context is used.
        if self.use_cide:
            self.cide_module = CIDE(emb_dim=emb_dim)
        else:
            self.cide_module = None

        # Diffusion UNet backbone (feature extractor at fixed t=1).
        self.unet = DiffusionUNet(
            in_channels=512, out_channels=4, model_channels=32,
            channel_mult=(1, 2, 4, 4), num_res_blocks=2,
            attention_resolutions=(4, 2, 1), num_heads=8,
            context_dim=emb_dim, transformer_depth=1,
        )

        self._init_aggregator_weights()

    def _init_aggregator_weights(self):
        for m in [self.layer1, self.layer2, self.out_layer]:
            for sub in m.modules():
                if isinstance(sub, (nn.Conv2d, nn.Linear)):
                    nn.init.trunc_normal_(sub.weight, std=0.02)
                    if sub.bias is not None:
                        nn.init.constant_(sub.bias, 0)

    def forward(self, audio_spec, rep_gated, gated_em, audio_wave=None):
        """
        audio_spec : (B, spec_in_channels, 128, 128)
        rep_gated  : (B, K, 4)         already gate-multiplied
        gated_em   : (B, 1, H, W) | None  used by 'input' mode
        audio_wave : (B, 2, T) | None  binaural waveform for CIDE (use_cide=True)
        """
        B = audio_spec.shape[0]

        if self.foa_mode == 'input':
            # The caller already concatenated gated_em into audio_spec; here we
            # just need a context for the cross-attention layers.
            context = self.null_context.expand(B, -1, -1)  # (B, 1, emb_dim)
        else:
            context = self.bin_proj(rep_gated,
                                    torch.ones(self.K, device=audio_spec.device))
            # rep_gated is already × g; pass g=1 to avoid double-gating.

        # Prepend CIDE token to the cross-attention context if enabled.
        # 'input' mode replaces the null token; 'condition' mode prepends
        # alongside the K bin tokens → (B, 1+K, emb_dim).
        if self.cide_module is not None:
            if audio_wave is None:
                raise ValueError(
                    "EchoDiffusionAmbi was constructed with use_cide=True "
                    "but forward() received audio_wave=None. Set "
                    "cfg.dataset.use_waveform=True so the dataloader returns "
                    "the 6-tuple including the binaural waveform.")
            cide_token = self.cide_module(audio_wave)        # (B, 1, emb_dim)
            if self.foa_mode == 'input':
                context = cide_token                          # replace null
            else:
                context = torch.cat([cide_token, context], dim=1)  # (B, 1+K, D)

        latents = self.aspp_asff(audio_spec)            # (B, 128, 32, 32)
        latents = self.latent_proj(latents)             # (B, 512, 32, 32)

        t = torch.ones((B,), device=audio_spec.device).long()
        outs = self.unet(latents, t, context=context)

        feats = [
            outs[0],
            outs[1],
            torch.cat([outs[2], F.interpolate(outs[3], scale_factor=2)], dim=1),
        ]
        x = torch.cat([self.layer1(feats[0]), self.layer2(feats[1]), feats[2]],
                      dim=1)
        return self.out_layer(x)


class EchoDiffusionAmbi(nn.Module):
    """EchoDiffusion + bin-gated FOA conditioning. See module docstring."""

    def __init__(self, cfg, max_depth=10.0, embed_dim=192, emb_dim=768,
                 K=8, foa_mode='condition', gate_init=2.0, gate_mask=None,
                 use_cide=False,
                 **_unused):
        super().__init__()
        self.max_depth = float(max_depth)
        self.K = int(K)
        self.foa_mode = str(foa_mode)
        self.use_cide = bool(use_cide)
        assert self.foa_mode in ('input', 'condition'), \
            f"foa_mode must be 'input' or 'condition', got {foa_mode!r}"

        spec_in_channels = 3 if self.foa_mode == 'input' else 2

        # ---------- Bin gate (same semantics as n4_0425) ----------
        if gate_mask is None:
            self.gate_logit = nn.Parameter(torch.full((self.K,), float(gate_init)))
            self.gate_learnable = True
        else:
            mask = torch.tensor(list(gate_mask), dtype=torch.float32)
            assert mask.numel() == self.K, \
                f"gate_mask length {mask.numel()} != K={self.K}"
            mask_c = mask.clamp(1e-6, 1 - 1e-6)
            logit = torch.log(mask_c / (1 - mask_c))
            self.register_buffer('gate_logit', logit, persistent=True)
            self.gate_learnable = False

        # FOA basis at the input ERP resolution. Used by 'input' mode to build
        # the gated energy map. ASPP+ASFF then resizes the (binaural+em) stack
        # internally to its 128×128 working resolution.
        H_erp, W_erp = (int(v) for v in cfg.dataset.images_size)
        basis = make_foa_basis_erp(H_erp, W_erp)         # (4, H, W)
        self.register_buffer('foa_basis', basis, persistent=False)

        # ---------- Diffusion-UNet feature extractor ----------
        channels_in = embed_dim * 8                       # 1536
        channels_out = embed_dim                          # 192
        self.encoder = EcoDepthEncoderAmbi(
            K=self.K, foa_mode=self.foa_mode,
            spec_in_channels=spec_in_channels,
            out_dim=channels_in, emb_dim=emb_dim,
            use_cide=self.use_cide,
        )
        self.decoder = Decoder(channels_in, channels_out)
        self.last_layer_depth = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, 3, 1, 1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels_out, 1, 3, 1, 1),
        )
        print(f"  [echodiff_ambi] foa_mode={self.foa_mode} K={self.K} "
              f"spec_in={spec_in_channels} use_cide={self.use_cide} "
              f"gate_learnable={self.gate_learnable}")

    def forward(self, x: torch.Tensor, rep_gt: torch.Tensor = None,
                audio_wave: torch.Tensor = None,
                **_unused) -> dict:
        """
        x      : (B, 2, H, W) binaural spectrogram (ERP resolution).
        rep_gt : (B, K, 4)    oracle per-bin FOA representatives (REQUIRED).
        """
        if rep_gt is None:
            raise ValueError(
                "EchoDiffusionAmbi requires rep_gt. Set "
                "cfg.dataset.use_distance_bins=True so the dataset returns "
                "the 5-tuple (audio, depth, foa_target, energy_map, rep_gt).")

        B, _, H, W = x.shape
        g = torch.sigmoid(self.gate_logit)                # (K,)
        rep_gated = rep_gt * g.view(1, -1, 1)             # (B, K, 4)

        # 1. Build the spec input (mode-dependent) ----------------------------
        if self.foa_mode == 'input':
            gated_em = _build_gated_em(rep_gt, g, self.foa_basis)  # (B,1,H,W)
            spec_input = torch.cat([x, gated_em], dim=1)           # (B,3,H,W)
        else:
            gated_em = None
            spec_input = x                                          # (B,2,H,W)

        # ASPP+ASFF expects 128×128. Match EchoDiffusion's pre-resize.
        if spec_input.shape[2] != 128 or spec_input.shape[3] != 128:
            spec_input = F.interpolate(spec_input, size=(128, 128),
                                       mode='bilinear', align_corners=False)

        # 2. Encoder + decoder + depth head ----------------------------------
        feats = self.encoder(spec_input, rep_gated, gated_em,
                             audio_wave=audio_wave)
        out = self.decoder(feats)
        out_depth = torch.sigmoid(self.last_layer_depth(out)) * self.max_depth
        if out_depth.shape[2] != H or out_depth.shape[3] != W:
            out_depth = F.interpolate(out_depth, size=(H, W), mode='nearest')

        return {
            'pred_depth': out_depth,
            'gate':       g,
            # Oracle model: rep_pred IS rep_gt. weighted_rep_loss is then 0,
            # but the key must be present so the train-step routing works.
            'rep_pred':   rep_gt,
            'pred_sh':    rep_gt[:, 0, :].contiguous(),
        }


if __name__ == '__main__':
    from types import SimpleNamespace as NS
    cfg = NS(dataset=NS(images_size=[256, 512], depth_norm=True))
    for mode in ('input', 'condition'):
        net = EchoDiffusionAmbi(cfg, K=8, foa_mode=mode).eval()
        x = torch.randn(2, 2, 256, 512)
        rep = torch.randn(2, 8, 4)
        with torch.no_grad():
            out = net(x, rep_gt=rep)
        n_total = sum(p.numel() for p in net.parameters()) / 1e6
        n_train = sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6
        print(f'mode={mode:10s} pred_depth={tuple(out["pred_depth"].shape)} '
              f'rep_pred={tuple(out["rep_pred"].shape)} '
              f'params={n_total:.2f}M (trainable {n_train:.2f}M)')
