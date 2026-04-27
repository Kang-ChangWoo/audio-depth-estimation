"""Energy-map-only depth estimation (exp247-256).

No binaural spectrogram — only the FOA energy map (or temporal energy maps)
as input. Tests whether the spatial energy distribution alone can predict depth.

Three 1→3 channel adaptor strategies for pretrained backbones:
  - repeat:  tile the single channel 3× → (3, H, W)
  - conv:    learned Conv2d(1, 3, 1) projection
  - edge:    (energy, grad_x, grad_y) → (3, H, W) — physics-informed

Two backbone families:
  - UNet (FOA0415V1Generator-based, input_nc=3)
  - ViT  (ViTFOABackbone-based, input_nc=3)

Two temporal modes:
  - temporal: 3 disjoint bins → (3, H, W) directly, no adaptor needed
  - temporal_ov: 3 overlapping bins → (3, H, W) directly
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.foa_0415_v1 import FOA0415V1Generator, SHHead
from models.pretrain.pretrained_vit_foa import ViTFOABackbone, SHHead as ViTSHHead


class RepeatAdaptor(nn.Module):
    """Repeat single channel 3 times."""
    def forward(self, x):
        return x.repeat(1, 3, 1, 1)


class ConvAdaptor(nn.Module):
    """Learned 1×1 conv: 1ch → 3ch."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 3, kernel_size=1)
    def forward(self, x):
        return self.conv(x)


class EdgeAdaptor(nn.Module):
    """(energy, sobel_x, sobel_y) → 3ch. Encodes spatial gradients."""
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                          dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                          dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)

    def forward(self, x):
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        return torch.cat([x, gx, gy], dim=1)


ADAPTORS = {
    'repeat': RepeatAdaptor,
    'conv':   ConvAdaptor,
    'edge':   EdgeAdaptor,
}


# ── UNet variants ─────────────────────────────────────────────

class EmapUNetGenerator(FOA0415V1Generator):
    """UNet that takes energy map (1ch) → adaptor → 3ch UNet."""

    DEFAULT_SH_DIM = 4
    DEFAULT_HEAD_HIDDEN = 256

    def __init__(self, cfg, input_nc=1, output_nc=1, num_downs=8, ngf=64,
                 use_dropout=False, sh_dim=None, head_hidden=None, **_unused):
        adaptor_name = getattr(cfg.model, 'adaptor', 'repeat')
        super().__init__(cfg, input_nc=3, output_nc=output_nc,
                         num_downs=num_downs, ngf=ngf,
                         use_dropout=use_dropout, sh_dim=sh_dim,
                         head_hidden=head_hidden)
        self.adaptor = ADAPTORS[adaptor_name]()

    def forward(self, x, **_unused):
        x = self.adaptor(x)
        return super().forward(x)


class EmapUNetTemporalGenerator(FOA0415V1Generator):
    """UNet that takes 3 temporal energy maps directly as 3ch input."""

    DEFAULT_SH_DIM = 4
    DEFAULT_HEAD_HIDDEN = 256

    def __init__(self, cfg, input_nc=3, output_nc=1, num_downs=8, ngf=64,
                 use_dropout=False, sh_dim=None, head_hidden=None, **_unused):
        super().__init__(cfg, input_nc=3, output_nc=output_nc,
                         num_downs=num_downs, ngf=ngf,
                         use_dropout=use_dropout, sh_dim=sh_dim,
                         head_hidden=head_hidden)


# ── ViT variants ──────────────────────────────────────────────

class EmapViTGenerator(nn.Module):
    """ViT-B/16 that takes energy map (1ch) → adaptor → 3ch ViT."""

    DEFAULT_SH_DIM = 4
    DEFAULT_HEAD_HIDDEN = 256

    def __init__(self, cfg, input_nc=1, pretrained=True, freeze_encoder=False,
                 sh_dim=None, head_hidden=None, **_unused):
        super().__init__()
        adaptor_name = getattr(cfg.model, 'adaptor', 'repeat')
        self.adaptor = ADAPTORS[adaptor_name]()
        self.sh_dim = int(sh_dim if sh_dim is not None else self.DEFAULT_SH_DIM)
        head_hidden = int(head_hidden if head_hidden is not None else self.DEFAULT_HEAD_HIDDEN)

        self.backbone = ViTFOABackbone(cfg, input_nc=3, pretrained=pretrained,
                                       freeze_encoder=freeze_encoder)
        self.sh_head = ViTSHHead(feat_dim=ViTFOABackbone.EMBED_DIM,
                                 sh_dim=self.sh_dim, hidden=head_hidden)

    def forward(self, x, **_unused):
        x = self.adaptor(x)
        tokens = self.backbone._encode(x)
        patch_tokens = tokens[:, 1:]
        cls_token = tokens[:, 0]
        pool = patch_tokens.mean(dim=1)
        sh_input = cls_token + pool
        pred_sh = self.sh_head(sh_input)
        pred_depth = self.backbone._decode_depth(
            patch_tokens, self.backbone.target_h, self.backbone.target_w)
        return {"pred_depth": pred_depth, "pred_sh": pred_sh}


class EmapViTTemporalGenerator(nn.Module):
    """ViT-B/16 that takes 3 temporal energy maps directly as 3ch input."""

    DEFAULT_SH_DIM = 4
    DEFAULT_HEAD_HIDDEN = 256

    def __init__(self, cfg, input_nc=3, pretrained=True, freeze_encoder=False,
                 sh_dim=None, head_hidden=None, **_unused):
        super().__init__()
        self.sh_dim = int(sh_dim if sh_dim is not None else self.DEFAULT_SH_DIM)
        head_hidden = int(head_hidden if head_hidden is not None else self.DEFAULT_HEAD_HIDDEN)

        self.backbone = ViTFOABackbone(cfg, input_nc=3, pretrained=pretrained,
                                       freeze_encoder=freeze_encoder)
        self.sh_head = ViTSHHead(feat_dim=ViTFOABackbone.EMBED_DIM,
                                 sh_dim=self.sh_dim, hidden=head_hidden)

    def forward(self, x, **_unused):
        tokens = self.backbone._encode(x)
        patch_tokens = tokens[:, 1:]
        cls_token = tokens[:, 0]
        pool = patch_tokens.mean(dim=1)
        pred_sh = self.sh_head(cls_token + pool)
        pred_depth = self.backbone._decode_depth(
            patch_tokens, self.backbone.target_h, self.backbone.target_w)
        return {"pred_depth": pred_depth, "pred_sh": pred_sh}
