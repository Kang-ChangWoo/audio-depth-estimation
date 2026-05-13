"""EchoRangeDepth — EchoDiffusion encoder/decoder + pluggable depth head.

Three head modes are selected via ``depth_head_type``:

  scalar   exact reproduction of the original EchoDiffusion scalar output
           (sigmoid × max_depth on a 3-conv head).
  range    softmax distribution over Br log-spaced depth bins; depth is
           recovered as the bin expectation (default) or median.
  hazard   per-bin first-hit hazards α_j; depth is recovered by the
           differentiable first-hit renderer (see HazardRangeDepthHead).

In all three modes the forward returns a dict with at least ``pred_depth``,
so the surrounding training/eval code can stay head-agnostic. The non-scalar
heads are stored as ``self.range_head`` regardless of which one is active,
so the ``range_bins`` buffer access pattern that the train loop uses
(``model.module.range_head.range_bins``) keeps working.

Forward output dict:

    scalar   {'pred_depth': (B, 1, H, W)}
    range    {'pred_depth', 'range_logits', 'range_prob',
              'range_entropy', 'range_bins'}
    hazard   {'pred_depth', 'range_logits', 'range_bins',
              'hazard_alpha', 'hazard_weights', 'hazard_bg_weight',
              'range_entropy', 'free_length'}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.echodiffusion.echodiffusion import EcoDepthEncoder, Decoder
from .range_head import RangeDepthHead, HazardRangeDepthHead


# Single source of truth for the supported head types — used both by this
# class and by argparse choices in train.py / test.py.
SUPPORTED_HEAD_TYPES = ("scalar", "range", "hazard")


class EchoRangeDepth(nn.Module):
    """EchoDiffusion backbone with a switchable depth head.

    Parameters mirror EchoDiffusion (same encoder, decoder, capacities)
    so checkpoint hyperparameters carry over for the scalar baseline.

    Parameters
    ----------
    max_depth                : float — for scalar head's sigmoid×max_depth
                                       AND for the hazard head's background
                                       depth term (w_bg * max_depth).
    embed_dim, emb_dim       : same as EchoDiffusion (192 / 768 by default).
    depth_head_type          : 'scalar' (default) | 'range' | 'hazard'.
    range_num_bins           : int — Br when range / hazard head is active.
    range_bin_spacing        : 'log' (default) or 'linear'.
    range_min_depth          : float — r_min for bin grid (metres).
    range_max_depth          : float — r_max for bin grid (metres).
    range_output_mode        : 'expectation' (default) or 'median'  (range only).
    hazard_bias_init         : float — initial bias of the hazard head's
                                       final 1×1 conv. Default −4.6 →
                                       sigmoid(−4.6) ≈ 0.01, so the renderer
                                       starts close to "everything is free
                                       space" and is shaped almost entirely
                                       by L_hit / L_free during warmup.
    """

    def __init__(self,
                 max_depth: float = 10.0,
                 embed_dim: int = 192,
                 emb_dim: int = 768,
                 depth_head_type: str = "scalar",
                 range_num_bins: int = 64,
                 range_bin_spacing: str = "log",
                 range_min_depth: float = 0.1,
                 range_max_depth: float = 20.0,
                 range_output_mode: str = "expectation",
                 hazard_bias_init: float = -4.6,
                 **_unused):
        super().__init__()
        self.max_depth = float(max_depth)
        self.depth_head_type = str(depth_head_type)
        if self.depth_head_type not in SUPPORTED_HEAD_TYPES:
            raise ValueError(
                f"depth_head_type must be one of {SUPPORTED_HEAD_TYPES}, "
                f"got {depth_head_type!r}")

        channels_in = embed_dim * 8                       # 1536
        channels_out = embed_dim                          # 192

        self.encoder = EcoDepthEncoder(out_dim=channels_in, emb_dim=emb_dim)
        self.decoder = Decoder(channels_in, channels_out)

        if self.depth_head_type == "scalar":
            self.scalar_head = nn.Sequential(
                nn.Conv2d(channels_out, channels_out, 3, 1, 1),
                nn.ReLU(inplace=False),
                nn.Conv2d(channels_out, 1, 3, 1, 1),
            )
            self.range_head = None
        elif self.depth_head_type == "range":
            self.scalar_head = None
            self.range_head = RangeDepthHead(
                in_channels=channels_out,
                num_bins=int(range_num_bins),
                r_min=float(range_min_depth),
                r_max=float(range_max_depth),
                spacing=range_bin_spacing,
                output_mode=range_output_mode,
            )
        else:  # 'hazard'
            self.scalar_head = None
            self.range_head = HazardRangeDepthHead(
                in_channels=channels_out,
                num_bins=int(range_num_bins),
                r_min=float(range_min_depth),
                r_max=float(range_max_depth),
                spacing=range_bin_spacing,
                max_depth=float(max_depth),
                bias_init=float(hazard_bias_init),
            )

    def forward(self, audio_spec: torch.Tensor,
                audio_wave: torch.Tensor, **_unused) -> dict:
        orig_h, orig_w = audio_spec.shape[2], audio_spec.shape[3]
        if orig_h != 128 or orig_w != 128:
            audio_spec = F.interpolate(audio_spec, size=(128, 128),
                                       mode='bilinear', align_corners=False)

        conv_feats = self.encoder(audio_spec, audio_wave)
        feat = self.decoder(conv_feats)                   # (B, 192, h, w)

        if self.depth_head_type == "scalar":
            pred = torch.sigmoid(self.scalar_head(feat)) * self.max_depth
            if pred.shape[2] != orig_h or pred.shape[3] != orig_w:
                pred = F.interpolate(pred, size=(orig_h, orig_w),
                                     mode='nearest')
            return {"pred_depth": pred}

        # range / hazard head — both expose pred_depth + range_entropy
        # at the decoder feature resolution; only those two need to be
        # upsampled to the input ERP resolution. Per-bin tensors stay at
        # the lower resolution to keep memory bounded.
        out = self.range_head(feat)
        if out["pred_depth"].shape[2] != orig_h or \
                out["pred_depth"].shape[3] != orig_w:
            out["pred_depth"] = F.interpolate(
                out["pred_depth"], size=(orig_h, orig_w), mode='nearest')
            out["range_entropy"] = F.interpolate(
                out["range_entropy"], size=(orig_h, orig_w), mode='nearest')
        return out


if __name__ == '__main__':
    # Lightweight shape check (no GPU).
    x = torch.randn(1, 2, 128, 128)
    w = torch.randn(1, 2, 5648)
    for ht in SUPPORTED_HEAD_TYPES:
        net = EchoRangeDepth(depth_head_type=ht,
                             range_num_bins=32,
                             range_min_depth=0.1,
                             range_max_depth=10.0).eval()
        with torch.no_grad():
            o = net(x, w)
        print(f"{ht:8s}:",
              {k: tuple(v.shape) if hasattr(v, 'shape') else v
               for k, v in o.items()})
