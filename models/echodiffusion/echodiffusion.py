"""EchoDiffusion: Depth estimation from binaural echoes using a diffusion UNet backbone.

Reproduces the architecture from:
  "EchoDiffusion: Continuous-time Diffusion Model for Audio-driven Depth Estimation"
  (Zhang et al.)

The model does NOT use iterative diffusion at inference. Instead, it repurposes
a diffusion UNet as a feature extractor with:
  - Audio spectrogram -> ASPP+ASFF UNet -> latent input (replaces noisy image)
  - Audio waveform -> Wav2Vec2 + CIDE -> cross-attention context (replaces text)
  - Diffusion UNet forward with t=1 -> hierarchical features
  - Feature aggregation + Decoder -> depth map
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from .aspp_asff import UNetASPPASFF
from .diffusion_unet import DiffusionUNet


class EmbeddingAdapter(nn.Module):
    """MLP residual adapter with learnable gamma scaling."""

    def __init__(self, emb_dim=768):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, texts, gamma):
        emb_transformed = self.fc(texts)
        texts = texts + gamma * emb_transformed
        texts = repeat(texts, 'n c -> n b c', b=1)
        return texts


class CIDE(nn.Module):
    """Conditioning via Information-Dense Embedding.

    Processes raw binaural waveform through:
    1. Conv1d to merge 2 channels to 1
    2. Wav2Vec2 (frozen) for audio feature extraction
    3. FC layers -> 100-class scene probabilities
    4. Softmax-weighted combination of learned embeddings
    5. EmbeddingAdapter (MLP + residual with learnable gamma)
    """

    def __init__(self, emb_dim=768):
        super().__init__()
        self.dim = emb_dim

        try:
            from transformers import Wav2Vec2Model
            self.wav2vec = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')
            self.wav2vec.freeze_feature_extractor()
            self._has_wav2vec = True
        except (ImportError, OSError):
            print("[CIDE] Warning: Wav2Vec2 not available, using random projection fallback")
            self.wav2vec_fallback = nn.Sequential(
                nn.Linear(960, 768),
                nn.GELU(),
                nn.Linear(768, 768),
            )
            self._has_wav2vec = False

        self.conv = nn.Conv1d(2, 1, kernel_size=1)
        self.fc = nn.Sequential(nn.Linear(768, 400), nn.GELU(), nn.Linear(400, 100))
        self.m = nn.Softmax(dim=1)
        self.embeddings = nn.Parameter(torch.randn(100, self.dim))
        self.embedding_adapter = EmbeddingAdapter(emb_dim=self.dim)
        self.gamma = nn.Parameter(torch.ones(self.dim) * 1e-4)

    def forward(self, x):
        # x: (B, 2, T) raw waveform
        x = self.conv(x).squeeze(1)  # (B, T)

        if self._has_wav2vec:
            # Ensure minimum length for Wav2Vec2
            min_len = 400
            if x.shape[1] < min_len:
                x = F.pad(x, (0, min_len - x.shape[1]))
            with torch.no_grad():
                wav2vec_output = self.wav2vec(x).last_hidden_state  # (B, T', 768)
            wav2vec_output = wav2vec_output.mean(dim=1)  # (B, 768)
        else:
            if x.shape[1] < 960:
                x = F.pad(x, (0, 960 - x.shape[1]))
            wav2vec_output = self.wav2vec_fallback(x[:, :960])

        class_probs = self.m(self.fc(wav2vec_output))      # (B, 100)
        class_embeddings = class_probs @ self.embeddings     # (B, 768)
        conditioning = self.embedding_adapter(class_embeddings, self.gamma)
        return conditioning


class EcoDepthEncoder(nn.Module):
    """Encoder: ASPP+ASFF spectrogram encoder -> Diffusion UNet feature extractor."""

    def __init__(self, out_dim=1536, ldm_prior=(32, 64, 256), emb_dim=768):
        super().__init__()

        # Feature aggregation layers for multi-scale UNet outputs
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

        # Audio conditioning
        self.cide_module = CIDE(emb_dim)

        # Spectrogram encoder -> 128ch 32x32 latent
        self.aspp_asff = UNetASPPASFF(in_channels=2, base_c=64)

        # Latent projection: 128ch -> 512ch for UNet input
        self.latent_proj = nn.Sequential(
            nn.Conv2d(128, 512, 1),
            nn.GroupNorm(32, 512),
            nn.SiLU(),
        )

        # Diffusion UNet backbone (feature extractor)
        self.unet = DiffusionUNet(
            in_channels=512, out_channels=4, model_channels=32,
            channel_mult=(1, 2, 4, 4), num_res_blocks=2,
            attention_resolutions=(4, 2, 1), num_heads=8,
            context_dim=emb_dim, transformer_depth=1,
        )

        self._init_weights()

    def _init_weights(self):
        for m in [self.layer1, self.layer2, self.out_layer]:
            for sub in m.modules():
                if isinstance(sub, (nn.Conv2d, nn.Linear)):
                    nn.init.trunc_normal_(sub.weight, std=0.02)
                    if sub.bias is not None:
                        nn.init.constant_(sub.bias, 0)

    def forward(self, audio_spec, audio_wave):
        # Encode spectrogram to latent
        latents = self.aspp_asff(audio_spec)           # (B, 128, 32, 32)
        latents = self.latent_proj(latents)             # (B, 512, 32, 32)

        # Encode waveform to cross-attention conditioning
        conditioning = self.cide_module(audio_wave)     # (B, 1, 768)

        # Fixed timestep t=1
        t = torch.ones((audio_spec.shape[0],), device=audio_spec.device).long()

        # UNet forward -> hierarchical features [finest, ..., coarsest]
        outs = self.unet(latents, t, context=conditioning)

        # Aggregate multi-scale features
        feats = [
            outs[0],
            outs[1],
            torch.cat([outs[2], F.interpolate(outs[3], scale_factor=2)], dim=1),
        ]

        x = torch.cat([self.layer1(feats[0]), self.layer2(feats[1]), feats[2]], dim=1)
        return self.out_layer(x)


class Decoder(nn.Module):
    """Deconvolution decoder with bilinear upsampling."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.deconv_layers = self._make_deconv_layer(3, [32, 32, 32], [2, 2, 2],
                                                      in_channels)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(32, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        out = self.deconv_layers(x)
        out = self.conv_layers(out)
        out = self.up(self.up(out))
        return out

    @staticmethod
    def _make_deconv_layer(num_layers, num_filters, num_kernels, in_planes):
        layers = []
        for i in range(num_layers):
            kernel = num_kernels[i]
            planes = num_filters[i]
            padding = 1 if kernel == 3 else (1 if kernel == 4 else 0)
            output_padding = 1 if kernel == 3 else 0
            layers.extend([
                nn.ConvTranspose2d(in_planes, planes, kernel, stride=2,
                                   padding=padding, output_padding=output_padding, bias=False),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True),
            ])
            in_planes = planes
        return nn.Sequential(*layers)


class EchoDiffusion(nn.Module):
    """EchoDiffusion: Audio-driven depth estimation using diffusion UNet backbone.

    Input:
        audio_spec: (B, 2, 128, 128) binaural spectrogram
        audio_wave: (B, 2, T) raw binaural waveform
    Output:
        depth: (B, 1, H, W) predicted depth map
    """

    def __init__(self, max_depth=10.0, embed_dim=192, emb_dim=768):
        super().__init__()
        self.max_depth = max_depth

        channels_in = embed_dim * 8   # 1536
        channels_out = embed_dim       # 192

        self.encoder = EcoDepthEncoder(out_dim=channels_in, emb_dim=emb_dim)
        self.decoder = Decoder(channels_in, channels_out)
        self.last_layer_depth = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, 3, 1, 1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels_out, 1, 3, 1, 1),
        )

    def forward(self, audio_spec, audio_wave):
        orig_h, orig_w = audio_spec.shape[2], audio_spec.shape[3]
        # Resize spectrogram to 128x128 for ASPP+ASFF UNet
        if orig_h != 128 or orig_w != 128:
            audio_spec = F.interpolate(audio_spec, size=(128, 128), mode='bilinear',
                                       align_corners=False)
        conv_feats = self.encoder(audio_spec, audio_wave)
        out = self.decoder(conv_feats)
        out_depth = torch.sigmoid(self.last_layer_depth(out)) * self.max_depth
        # Resize output to match original input spatial dimensions
        if out_depth.shape[2] != orig_h or out_depth.shape[3] != orig_w:
            out_depth = F.interpolate(out_depth, size=(orig_h, orig_w), mode='nearest')
        return out_depth
