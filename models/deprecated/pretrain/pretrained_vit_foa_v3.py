"""v3 — Early-layer SH tap conditioning the depth decoder via FiLM.

Rationale
---------
Directional cues in the ambisonic signal are *low-level* features: they
live in inter-channel phase / ILD / early-reflection timing, which are
captured best by the *early* ViT blocks before the network abstracts
them away. Late blocks are where global depth semantics are built up.
v3 splits those roles:

  blocks[0 : N_early] ──► mean-pool ──► SHHead ──► pred_sh
                                         │
                                         └─► FiLM(gamma, beta)
                                                    │
  blocks[N_early : end] ──► patch tokens ──► (FiLM modulated)
                                          ──► DepthDecoder ──► depth

So the SH prediction is built from early spatial features, and the
resulting latent is used to *condition* the final depth decoder — the
FOA information explicitly shapes depth.

Loss routing is the v1 aux-SH-head path (``_train_step_foa_0415``):
    L = L_depth + lambda_sh * L1(pred_sh[:, :4], gt_foa)
"""

import torch
import torch.nn as nn

from .pretrained_vit_foa import ViTFOABackbone, SHHead


class FiLMProjector(nn.Module):
    """Projects an SH latent into per-channel (gamma, beta) for FiLM."""

    def __init__(self, in_dim: int, channels: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * channels),
        )
        self.channels = channels

    def forward(self, latent: torch.Tensor):
        gb = self.net(latent)
        gamma, beta = gb.chunk(2, dim=-1)
        return gamma, beta


class PretrainedViTFOAV3(ViTFOABackbone):

    DEFAULT_SH_DIM = 4
    DEFAULT_HEAD_HIDDEN = 256
    DEFAULT_N_EARLY = 4            # blocks [0:4] feed the SH head

    def __init__(self, cfg, input_nc=2, pretrained=True, freeze_encoder=False,
                 sh_dim: int = None, head_hidden: int = None,
                 n_early: int = None, **_unused):
        super().__init__(cfg, input_nc=input_nc, pretrained=pretrained,
                         freeze_encoder=freeze_encoder)
        self.sh_dim = int(sh_dim if sh_dim is not None else self.DEFAULT_SH_DIM)
        hidden = int(head_hidden if head_hidden is not None
                     else self.DEFAULT_HEAD_HIDDEN)
        self.n_early = int(n_early if n_early is not None else self.DEFAULT_N_EARLY)
        assert 1 <= self.n_early < self.NUM_LAYERS, \
            f"n_early must be in [1, {self.NUM_LAYERS - 1}), got {self.n_early}"

        self.sh_head = SHHead(feat_dim=2 * self.EMBED_DIM, sh_dim=self.sh_dim,
                              hidden=hidden)
        self.film = FiLMProjector(in_dim=self.sh_dim, channels=self.EMBED_DIM,
                                  hidden=hidden)

    def _run_split(self, x):
        """Run embed + pos-emb, then blocks[0:n_early], SH, FiLM, blocks[n_early:]."""
        tokens = self._embed(x)
        tokens = tokens + self.encoder.pos_embedding
        tokens = self.encoder.dropout(tokens)

        for i in range(self.n_early):
            tokens = self.encoder.layers[i](tokens)

        # --- SH prediction from early tokens ---
        early_cls = tokens[:, 0]
        early_patch_mean = tokens[:, 1:].mean(dim=1)
        sh_feat = torch.cat([early_cls, early_patch_mean], dim=1)
        pred_sh = self.sh_head(sh_feat)                  # (B, sh_dim)

        # --- FiLM modulation injected into all remaining tokens ---
        gamma, beta = self.film(pred_sh)                 # (B, D), (B, D)
        gamma = gamma.unsqueeze(1)                       # (B, 1, D)
        beta = beta.unsqueeze(1)
        tokens = tokens * (1.0 + gamma) + beta           # broadcast over (1+N)

        for i in range(self.n_early, self.NUM_LAYERS):
            tokens = self.encoder.layers[i](tokens)
        tokens = self.encoder.ln(tokens)
        return tokens, pred_sh

    def forward(self, x, **_unused):
        orig_h, orig_w = x.shape[2], x.shape[3]
        tokens, pred_sh = self._run_split(x)
        patch_tokens = tokens[:, 1:]
        pred_depth = self._decode_depth(patch_tokens, orig_h, orig_w)
        return {"pred_depth": pred_depth, "pred_sh": pred_sh}
