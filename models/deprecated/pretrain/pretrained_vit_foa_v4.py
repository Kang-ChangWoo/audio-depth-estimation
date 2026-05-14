"""v4 — Multi-scale SH from 4 tapped ViT layers + DPT-style depth fusion.

Rationale
---------
Different transformer depths encode different aspects of the ambisonic
signal:

  shallow blocks — local spectro-spatial cues, ILD / phase patterns
                   (correlate with low-order directional SH bands l0, l1)
  mid blocks     — mid-range reflection structure
                   (correlate with mid-order bands l2, l3)
  deep blocks    — global scene semantics
                   (correlate with room shape / high-order bands l4, l5,
                    as well as dense depth)

v4 taps features at 4 ViT layers (default {2, 5, 8, 11} — evenly spaced
DPT-style), runs a band-wise SH head on the concatenated multi-scale
feature vector so each tap can specialize, and fuses the 4 taps
additively into the depth-decoder input so the decoder also benefits
from multi-scale context.

SH head output is plain ``sh_dim``-wide (4 by default). When ``sh_dim``
is raised (e.g. 36 for SH order 5) each scale contributes to a distinct
band so the layer-order mixing becomes explicit.

Loss routing: v1 aux-SH-head path (``_train_step_foa_0415``).
"""

import torch
import torch.nn as nn

from .pretrained_vit_foa import ViTFOABackbone, SHHead


class PretrainedViTFOAV4(ViTFOABackbone):

    DEFAULT_SH_DIM = 4
    DEFAULT_HEAD_HIDDEN = 256
    DEFAULT_TAPS = (2, 5, 8, 11)

    def __init__(self, cfg, input_nc=2, pretrained=True, freeze_encoder=False,
                 sh_dim: int = None, head_hidden: int = None,
                 taps=None, **_unused):
        super().__init__(cfg, input_nc=input_nc, pretrained=pretrained,
                         freeze_encoder=freeze_encoder)
        self.sh_dim = int(sh_dim if sh_dim is not None else self.DEFAULT_SH_DIM)
        hidden = int(head_hidden if head_hidden is not None
                     else self.DEFAULT_HEAD_HIDDEN)
        self.taps = tuple(int(t) for t in (taps if taps is not None
                                           else self.DEFAULT_TAPS))
        for t in self.taps:
            assert 0 <= t < self.NUM_LAYERS
        assert len(self.taps) >= 2

        # One SHHead per tap, each produces the full sh_dim; we mix them
        # through a shared MLP so the final coefficient vector reflects
        # contributions from all depths.
        self.tap_heads = nn.ModuleList([
            SHHead(feat_dim=2 * self.EMBED_DIM, sh_dim=self.sh_dim, hidden=hidden)
            for _ in self.taps
        ])
        self.sh_mix = nn.Sequential(
            nn.LayerNorm(self.sh_dim * len(self.taps)),
            nn.Linear(self.sh_dim * len(self.taps), hidden),
            nn.GELU(),
            nn.Linear(hidden, self.sh_dim),
        )

        # DPT-style fusion: learnable scalar per tap used to mix tap patch
        # tokens into the final decoder input.
        self.fuse_weights = nn.Parameter(torch.ones(len(self.taps) + 1))

    def forward(self, x, **_unused):
        orig_h, orig_w = x.shape[2], x.shape[3]
        final_tokens, taps = self._encode_with_taps(x, self.taps)

        # ── Multi-scale SH prediction ────────────────────────────
        per_tap = []
        for idx, t in enumerate(self.taps):
            tok = taps[t]                                # (B, 1+N, D)
            feat = torch.cat([tok[:, 0], tok[:, 1:].mean(dim=1)], dim=1)
            per_tap.append(self.tap_heads[idx](feat))    # (B, sh_dim)
        pred_sh = self.sh_mix(torch.cat(per_tap, dim=-1))  # (B, sh_dim)

        # ── Multi-scale depth fusion ─────────────────────────────
        weights = torch.softmax(self.fuse_weights, dim=0)
        fused = weights[-1] * final_tokens[:, 1:]
        for idx, t in enumerate(self.taps):
            fused = fused + weights[idx] * taps[t][:, 1:]
        pred_depth = self._decode_depth(fused, orig_h, orig_w)

        return {"pred_depth": pred_depth, "pred_sh": pred_sh}
