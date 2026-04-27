"""Implicit Sound-Field Projection Fusion.

Takes an intermediate feature F_in ∈ R^{B×C×H×W} from the binaural encoder,
produces (a) per-distance-bin FOA representatives rep_pred ∈ R^{B×8×4} for
Method-E supervision, (b) an ERP projection of those representatives
converted through a learned adapter to C channels, and (c) a gated
residual update of F_in. Returns (F_out, rep_pred).

Key design points (see Initial Model Implementation Plan):
  1. Implicit bin-wise latent z_sf ∈ R^{B×K×D} from a pooled F_in token.
  2. Method-E coefficient readout: z_sf → rep_pred. Supervised by rep_gt.
  3. ERP basis projection of rep_pred, both signed and log1p(signed²) maps,
     with per-bin learnable gates (near strong, far weak).
  4. Learned map_proj + FiLM adapter from K*D latent for depth-useful feature.
  5. Raw path + sigmoid gate + small residual scale — F_out stays close to F_in
     at init so the fusion doesn't destabilize an otherwise-working decoder.

FOA channel order is [W, Y, Z, X] (ACN). ERP convention matches
data/sh_basis.py: row 0 = top (elevation +π/2), row H-1 = bottom (-π/2),
azimuth in [-π, π] left to right.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inv_sigmoid(x, eps: float = 1e-6):
    x = torch.as_tensor(x, dtype=torch.float32).clamp(eps, 1.0 - eps)
    return torch.log(x / (1.0 - x))


def make_foa_basis_erp(H: int, W: int,
                       device=None, dtype=torch.float32) -> torch.Tensor:
    """FOA basis on an ERP grid, channel order [W, Y, Z, X]. Shape (4, H, W).

        basis[0] = c0                      (W, isotropic)
        basis[1] = c1 · cos(el) · sin(az)  (Y)
        basis[2] = c1 · sin(el)            (Z)
        basis[3] = c1 · cos(el) · cos(az)  (X)

    c0 = 1/√(4π),  c1 = √(3/(4π)).
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


class ImplicitSoundFieldProjectionFusion(nn.Module):
    """See module docstring."""

    # Near bins start strong, far bins weak. Logits are the inverse-sigmoid
    # of these so sigmoid(bin_gate_logit) ≈ DEFAULT_GATE at init.
    DEFAULT_GATE = (1.00, 1.00, 0.85, 0.70, 0.50, 0.35, 0.20, 0.15)

    def __init__(self, C: int, H: int, W: int,
                 K: int = 8, D: int = 128,
                 res_scale_init: float = 0.1,
                 gate_learnable: bool = True,
                 enable_fusion: bool = True):
        super().__init__()
        assert len(self.DEFAULT_GATE) == K, \
            f"DEFAULT_GATE length ({len(self.DEFAULT_GATE)}) != K ({K})"
        self.C, self.H, self.W, self.K, self.D = C, H, W, K, D
        self.enable_fusion = enable_fusion
        self.gate_learnable = gate_learnable

        # 1. Latent-token generator — pooled F_in → K bin-specific tokens.
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.to_sf_tokens = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(C),
            nn.Linear(C, K * D),
        )
        self.bin_embed = nn.Parameter(torch.randn(1, K, D) * 0.02)

        # 2. Method-E coefficient readout: D → 4 (W, Y, Z, X).
        self.coeff_head = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, 4),
        )

        # 3. Per-bin gate (learnable or fixed).
        init_gate_logit = _inv_sigmoid(self.DEFAULT_GATE).view(1, K, 1, 1)
        if gate_learnable:
            self.bin_gate_logit = nn.Parameter(init_gate_logit)
        else:
            # Persistent=False so the buffer stays on-device but isn't
            # expected in external state_dicts (checkpoints) either way.
            self.register_buffer('bin_gate_logit', init_gate_logit,
                                 persistent=True)

        # 4. Map adapter: 2K channels (signed + log1p-energy) → C.
        self.map_proj = nn.Sequential(
            nn.Conv2d(2 * K, C, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(C, C, kernel_size=1),
        )

        # 5. FiLM from pooled bin latent.
        self.latent_film = nn.Sequential(
            nn.LayerNorm(K * D),
            nn.Linear(K * D, 2 * C),
        )

        # 6. Raw path + gated residual fusion.
        self.raw_proj = nn.Conv2d(C, C, kernel_size=1)
        self.fuse_gate = nn.Sequential(
            nn.Conv2d(2 * C, C, kernel_size=1),
            nn.Sigmoid(),
        )
        self.res_scale = nn.Parameter(torch.tensor(float(res_scale_init)))

        # ERP basis at the *feature grid* resolution (H, W). The
        # energy_map_loss can use this same buffer when comparing rep_pred
        # and rep_gt projections.
        basis = make_foa_basis_erp(H, W)
        self.register_buffer('basis', basis, persistent=False)

    def forward(self, F_in: torch.Tensor):
        """F_in: (B, C, H, W) → (F_out, rep_pred).

        When ``enable_fusion=False``, returns (F_in, rep_pred) — i.e. the
        feature passes through unmodified. The rep_pred head still runs
        so Stage-B "aux only" experiments can supervise it.
        """
        B, C, H, W = F_in.shape
        assert C == self.C, f"C mismatch: got {C}, expected {self.C}"
        assert H == self.H and W == self.W, \
            f"HW mismatch: got {(H, W)}, expected {(self.H, self.W)}"

        # 1. Bin-wise latent.
        pooled = self.pool(F_in)                          # (B, C, 1, 1)
        z_sf = self.to_sf_tokens(pooled).view(B, self.K, self.D)
        z_sf = z_sf + self.bin_embed                      # (B, K, D)

        # 2. Coefficient readout.
        rep_pred = self.coeff_head(z_sf)                  # (B, K, 4)

        if not self.enable_fusion:
            return F_in, rep_pred

        # 3. ERP basis projection.
        signed = torch.einsum('bkc,chw->bkhw', rep_pred, self.basis)
        energy = torch.log1p(signed.square())

        bin_gate = torch.sigmoid(self.bin_gate_logit)     # (1, K, 1, 1)
        signed = signed * bin_gate
        energy = energy * bin_gate

        proj_map = torch.cat([signed, energy], dim=1)     # (B, 2K, H, W)

        # 4. Map adapter → C-channel feature.
        F_sf = self.map_proj(proj_map)                    # (B, C, H, W)

        # FiLM from pooled bin latents.
        z_flat = z_sf.flatten(1)                          # (B, K*D)
        film = self.latent_film(z_flat)                   # (B, 2C)
        gamma, beta = film.chunk(2, dim=-1)
        gamma = gamma.view(B, self.C, 1, 1)
        beta = beta.view(B, self.C, 1, 1)
        F_sf = F_sf * (1.0 + 0.1 * torch.tanh(gamma)) + 0.1 * beta

        # 5. Gated residual fusion. F_out ≈ F_raw at init (res_scale small).
        F_raw = self.raw_proj(F_in)
        gate = self.fuse_gate(torch.cat([F_raw, F_sf], dim=1))
        F_out = F_raw + self.res_scale * gate * F_sf

        return F_out, rep_pred
