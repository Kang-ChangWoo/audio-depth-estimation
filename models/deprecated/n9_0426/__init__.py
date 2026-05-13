"""n9_0426 — n9_0425 cascade with a pretrained-ViT/ResNet outer hourglass.

Same I/O contract as n9_0425. The outer 8-level UNet (binaural enc +
parallel EM enc + UNet decoder) is replaced by a pretrained ViT-B/16 or
ResNet-50 hourglass that consumes ``concat([binaural, gated_em], dim=1)``
as a 3-channel input. Inner FOA predictor is unchanged (frozen-or-
finetuned n3_0425).

Backbone selectable at construction via ``cfg.model.backbone``:
    'vit'    — torchvision vit_b_16 (ImageNet weights, interpolated pos embeds)
    'resnet' — torchvision resnet50 (ImageNet weights, FPN-style decoder)

Freeze knobs:
    cfg.model.freeze_backbone — freeze pretrained backbone weights
    cfg.model.freeze_n3       — freeze inner n3 (default True)
"""

from .n9_0426 import N9_0426Net

__all__ = ['N9_0426Net']
