"""EchoRangeDepth — EchoDiffusion encoder/decoder + pluggable depth head.

When ``depth_head_type='scalar'`` the model reproduces the original
EchoDiffusion scalar output exactly (sigmoid × max_depth on a 3-conv
head). When ``depth_head_type='range'`` the head is replaced by a
RangeDepthHead that outputs a per-pixel softmax distribution over
log-spaced depth bins; the predicted scalar depth is recovered as the
expectation (or median) of that distribution.

Forward returns a dict so downstream code is uniform across head types:

    {
        'pred_depth':     (B, 1, H, W),   # always present
        'range_logits':   (B, Br, H, W),  # range head only
        'range_prob':     (B, Br, H, W),  # range head only
        'range_entropy':  (B, 1, H, W),   # range head only
        'range_bins':     (Br,),          # range head only
    }
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.echodiffusion.echodiffusion import EcoDepthEncoder, Decoder
from .range_head import RangeDepthHead


class EchoRangeDepth(nn.Module):
    """EchoDiffusion backbone with a switchable depth head.

    Parameters mirror EchoDiffusion (same encoder, decoder, capacities)
    so checkpoint hyperparameters carry over for the scalar baseline.

    Parameters
    ----------
    max_depth                : float — for scalar head's sigmoid×max_depth.
    embed_dim, emb_dim       : same as EchoDiffusion (192 / 768 by default).
    depth_head_type          : 'scalar' (default — exact baseline) or 'range'.
    range_num_bins           : int — Br when range head is active.
    range_bin_spacing        : 'log' (default) or 'linear'.
    range_min_depth          : float — r_min for range bins (metres).
    range_max_depth          : float — r_max for range bins (metres).
    range_output_mode        : 'expectation' (default) or 'median'.
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
                 **_unused):
        super().__init__()
        self.max_depth = float(max_depth)
        self.depth_head_type = str(depth_head_type)

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
        else:
            raise ValueError(
                f"depth_head_type must be 'scalar' or 'range', "
                f"got {depth_head_type!r}")

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

        # range head
        out = self.range_head(feat)
        # Upsample only the scalar pred_depth to original spatial size to
        # avoid moving the (B, Br, h, w) logits tensor around at full res.
        if out["pred_depth"].shape[2] != orig_h or \
                out["pred_depth"].shape[3] != orig_w:
            out["pred_depth"] = F.interpolate(
                out["pred_depth"], size=(orig_h, orig_w), mode='nearest')
            out["range_entropy"] = F.interpolate(
                out["range_entropy"], size=(orig_h, orig_w), mode='nearest')
        return out


if __name__ == '__main__':
    # Lightweight shape check (no GPU).
    net_scalar = EchoRangeDepth(depth_head_type="scalar").eval()
    net_range = EchoRangeDepth(depth_head_type="range",
                               range_num_bins=64).eval()
    x = torch.randn(1, 2, 128, 128)
    w = torch.randn(1, 2, 5648)
    with torch.no_grad():
        s = net_scalar(x, w)
        r = net_range(x, w)
    print("scalar:", {k: tuple(v.shape) if hasattr(v, 'shape') else v
                      for k, v in s.items()})
    print("range :", {k: tuple(v.shape) if hasattr(v, 'shape') else v
                      for k, v in r.items()})
