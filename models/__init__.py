"""models package — keep-as-code modules only.

Phase E1 (2026-05-13): the unused n*_*, renew/, and foa_* trial modules
moved to models/deprecated/. This package now exposes only classes that
the keep-list configs (config/baseline.yaml, foa.yaml, vit.yaml,
batvision.yaml, echonet.yaml, echorange.yaml, pretrain_*.yaml) and the
echodiffusion-family subpackages actually reference.

unet.py and vit.py are retained as code because config/baseline.yaml
selects ``unet_baseline`` (-> define_G from .unet) and config/vit.yaml
selects ``vit_baseline`` (-> AudioDepthViT from .vit).
"""

from .unet import UnetGenerator, UnetSkipConnectionBlock, define_G
from .unet_foa import AudioDepthFOAGenerator, DeepScaleShift
from .echodiffusion import (
    EchoDiffusion, EchoDiffusionAmbi, EchoDiffusionAmbiSH,
    EchoDiffusionSHSidePlus,
)
from .bin_based import EchoRangeDepth, RangeDepthHead, soft_range_nll_loss
from .echonet import EchoNet
from .batvision import BatVisionUNet
from .pretrain import (
    PretrainedViT, PretrainedResNet,
    PretrainedViTFOA, PretrainedViTFOAV2, PretrainedViTFOAV3,
    PretrainedViTFOAV4, PretrainedViTFOAV5,
    PretrainedViTFOAV6EAttn, PretrainedViTFOAV6MSSH, PretrainedViTFOAV6OracleNC3,
)
from .vit import AudioDepthViT
from .losses import (
    SILogLoss, BerHuLoss, DepthLoss,
    FOAGuidedLoss, SHHistogramAlignmentLoss, AudioDepthFOALoss,
    KLDivRegLoss,
)
