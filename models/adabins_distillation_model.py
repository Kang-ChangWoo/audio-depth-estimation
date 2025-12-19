"""
AdaBins with Knowledge Distillation from RGB to Audio

Architecture:
    Training:
        - RGB Encoder (Teacher) → Predicts depth & features
        - Audio Encoder (Student) → Learns from RGB teacher
        - Distillation losses transfer knowledge
    
    Inference:
        - Audio Encoder only (completely independent)
        - No RGB input needed

Key Differences from Multi-Modal:
    - NO fusion of RGB and Audio features
    - Audio learns to mimic RGB's behavior
    - At inference, Audio works standalone
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from typing import Dict, Optional, Tuple


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class AdaBinsEncoder(nn.Module):
    """
    Encoder for AdaBins (can be RGB or Audio)
    """
    
    def __init__(self, input_channels, base_channels=64):
        super().__init__()
        self.inc = DoubleConv(input_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 8)
        
    def forward(self, x):
        """Returns multi-scale features"""
        x1 = self.inc(x)      # 64 channels
        x2 = self.down1(x1)   # 128 channels
        x3 = self.down2(x2)   # 256 channels
        x4 = self.down3(x3)   # 512 channels
        x5 = self.down4(x4)   # 512 channels (bottleneck)
        return {'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5}


class AdaBinsBinPredictor(nn.Module):
    """
    Predicts adaptive bin centers from global features
    """
    
    def __init__(self, bottleneck_dim=512, n_bins=128, max_depth=30.0):
        super().__init__()
        self.n_bins = n_bins
        self.max_depth = max_depth
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.predictor = nn.Sequential(
            nn.Linear(bottleneck_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, n_bins),
            nn.Softmax(dim=1)  # Bin widths (sum to 1)
        )
    
    def forward(self, features):
        """
        Args:
            features: [B, C, H, W] - Bottleneck features
        
        Returns:
            bin_centers: [B, n_bins] - Adaptive bin centers
            bin_widths: [B, n_bins] - Bin widths
        """
        global_feat = self.adaptive_pool(features).flatten(1)
        bin_widths = self.predictor(global_feat)
        
        # Cumulative sum to get bin edges
        bin_edges = torch.cumsum(bin_widths, dim=1)
        bin_edges = torch.cat([
            torch.zeros_like(bin_edges[:, :1]),
            bin_edges
        ], dim=1) * self.max_depth
        
        # Bin centers = midpoint of edges
        bin_centers = (bin_edges[:, :-1] + bin_edges[:, 1:]) / 2
        
        return bin_centers, bin_widths


class AdaBinsDecoder(nn.Module):
    """
    Decoder that predicts per-pixel bin classification
    """
    
    def __init__(self, base_channels=64, n_bins=128, output_size=256):
        super().__init__()
        self.n_bins = n_bins
        self.output_size = output_size
        
        # Decoder path
        # INPUT channels = concat(previous_output + skip_connection)
        # Encoder outputs: x1=64, x2=128, x3=256, x4=512, x5=512
        self.up1 = Up(1024, base_channels * 8, bilinear=True)  # 512+512 -> 512
        self.up2 = Up(768, base_channels * 4, bilinear=True)   # 512+256 -> 256
        self.up3 = Up(384, base_channels * 2, bilinear=True)   # 256+128 -> 128
        self.up4 = Up(192, base_channels, bilinear=True)       # 128+64 -> 64
        
        # Classification head
        self.class_head = nn.Conv2d(base_channels, n_bins, kernel_size=1)
    
    def forward(self, features, bin_centers):
        """
        Args:
            features: Dict with x1-x5 encoder features
            bin_centers: [B, n_bins] - Adaptive bin centers
        
        Returns:
            bin_logits: [B, n_bins, H, W]
            base_depth: [B, 1, H, W]
        """
        x5 = features['x5']
        x4 = features['x4']
        x3 = features['x3']
        x2 = features['x2']
        x1 = features['x1']
        
        # Decode
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Classification
        logits = self.class_head(x)
        
        if logits.shape[-1] != self.output_size:
            logits = F.interpolate(logits, size=(self.output_size, self.output_size),
                                  mode='bilinear', align_corners=False)
        
        # Soft binning: depth = Σ(prob_i * bin_center_i)
        probs = F.softmax(logits, dim=1)
        bin_centers_map = bin_centers[:, :, None, None]  # [B, n_bins, 1, 1]
        base_depth = (probs * bin_centers_map).sum(dim=1, keepdim=True)
        
        return logits, base_depth


class AdaBinsDistillationModel(nn.Module):
    """
    AdaBins with Knowledge Distillation from RGB to Audio
    
    Training:
        - RGB Teacher: Predicts depth from images
        - Audio Student: Learns to mimic RGB's predictions and features
        - Distillation losses transfer knowledge
    
    Inference:
        - Audio Student only (RGB not needed)
    
    Args:
        n_bins: Number of adaptive bins
        base_channels: Base channel count
        output_size: Output spatial resolution
        max_depth: Maximum depth value
        use_pretrained_rgb: Load pre-trained RGB encoder
    """
    
    def __init__(
        self,
        n_bins=128,
        base_channels=64,
        output_size=256,
        max_depth=30.0,
        use_pretrained_rgb=False
    ):
        super().__init__()
        self.n_bins = n_bins
        self.max_depth = max_depth
        self.output_size = output_size
        
        # ==========================================
        # RGB Teacher (3 channels)
        # ==========================================
        self.rgb_encoder = AdaBinsEncoder(input_channels=3, base_channels=base_channels)
        self.rgb_bin_predictor = AdaBinsBinPredictor(
            bottleneck_dim=base_channels * 8,
            n_bins=n_bins,
            max_depth=max_depth
        )
        self.rgb_decoder = AdaBinsDecoder(
            base_channels=base_channels,
            n_bins=n_bins,
            output_size=output_size
        )
        
        if use_pretrained_rgb:
            self._load_pretrained_rgb()
        
        # ==========================================
        # Audio Student (2 channels)
        # ==========================================
        self.audio_encoder = AdaBinsEncoder(input_channels=2, base_channels=base_channels)
        self.audio_bin_predictor = AdaBinsBinPredictor(
            bottleneck_dim=base_channels * 8,
            n_bins=n_bins,
            max_depth=max_depth
        )
        self.audio_decoder = AdaBinsDecoder(
            base_channels=base_channels,
            n_bins=n_bins,
            output_size=output_size
        )
        
        # ==========================================
        # Residual Refinement (Shared Architecture)
        # ==========================================
        # Both RGB and Audio can use residual refinement
        self.residual_head = nn.Conv2d(base_channels, 1, kernel_size=1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
    
    def _load_pretrained_rgb(self):
        """Load pre-trained RGB encoder"""
        print("INFO: Placeholder for loading pre-trained RGB encoder")
        # TODO: Implement loading from pre-trained depth estimation model
        pass
    
    def forward_rgb(self, rgb):
        """
        Forward pass for RGB (Teacher)
        
        Args:
            rgb: [B, 3, H, W]
        
        Returns:
            Dictionary with RGB predictions
        """
        # Encode
        rgb_feats = self.rgb_encoder(rgb)
        
        # Predict bins
        rgb_bin_centers, rgb_bin_widths = self.rgb_bin_predictor(rgb_feats['x5'])
        
        # Decode
        rgb_logits, rgb_base_depth = self.rgb_decoder(rgb_feats, rgb_bin_centers)
        
        # Residual (from final decoder features)
        # We need to get the final decoder output before classification head
        # For simplicity, use a separate forward pass through decoder
        x5 = rgb_feats['x5']
        x4 = rgb_feats['x4']
        x3 = rgb_feats['x3']
        x2 = rgb_feats['x2']
        x1 = rgb_feats['x1']
        
        x = self.rgb_decoder.up1(x5, x4)
        x = self.rgb_decoder.up2(x, x3)
        x = self.rgb_decoder.up3(x, x2)
        x = self.rgb_decoder.up4(x, x1)
        
        residual_raw = self.residual_head(x)
        if residual_raw.shape[-1] != self.output_size:
            residual_raw = F.interpolate(residual_raw, size=(self.output_size, self.output_size),
                                        mode='bilinear', align_corners=False)
        rgb_residual = torch.tanh(residual_raw) * (self.max_depth * 0.2)
        
        rgb_final_depth = torch.clamp(rgb_base_depth + rgb_residual, 0, self.max_depth)
        
        return {
            'features': rgb_feats,
            'bin_centers': rgb_bin_centers,
            'bin_widths': rgb_bin_widths,
            'bin_logits': rgb_logits,
            'base_depth': rgb_base_depth,
            'residual': rgb_residual,
            'final_depth': rgb_final_depth
        }
    
    def forward_audio(self, audio):
        """
        Forward pass for Audio (Student)
        
        Args:
            audio: [B, 2, H, W]
        
        Returns:
            Dictionary with Audio predictions
        """
        # Encode
        audio_feats = self.audio_encoder(audio)
        
        # Predict bins
        audio_bin_centers, audio_bin_widths = self.audio_bin_predictor(audio_feats['x5'])
        
        # Decode
        audio_logits, audio_base_depth = self.audio_decoder(audio_feats, audio_bin_centers)
        
        # Residual
        x5 = audio_feats['x5']
        x4 = audio_feats['x4']
        x3 = audio_feats['x3']
        x2 = audio_feats['x2']
        x1 = audio_feats['x1']
        
        x = self.audio_decoder.up1(x5, x4)
        x = self.audio_decoder.up2(x, x3)
        x = self.audio_decoder.up3(x, x2)
        x = self.audio_decoder.up4(x, x1)
        
        residual_raw = self.residual_head(x)
        if residual_raw.shape[-1] != self.output_size:
            residual_raw = F.interpolate(residual_raw, size=(self.output_size, self.output_size),
                                        mode='bilinear', align_corners=False)
        audio_residual = torch.tanh(residual_raw) * (self.max_depth * 0.2)
        
        audio_final_depth = torch.clamp(audio_base_depth + audio_residual, 0, self.max_depth)
        
        return {
            'features': audio_feats,
            'bin_centers': audio_bin_centers,
            'bin_widths': audio_bin_widths,
            'bin_logits': audio_logits,
            'base_depth': audio_base_depth,
            'residual': audio_residual,
            'final_depth': audio_final_depth
        }
    
    def forward(self, audio, rgb=None, mode='train'):
        """
        Forward pass
        
        Args:
            audio: [B, 2, H, W] - Audio input
            rgb: [B, 3, H, W] - RGB input (only for training)
            mode: 'train' or 'inference'
        
        Returns:
            Dictionary with predictions
        """
        # Always compute audio predictions
        audio_output = self.forward_audio(audio)
        
        # Compute RGB predictions only during training
        if mode == 'train' and rgb is not None:
            with torch.no_grad():  # RGB is frozen teacher
                rgb_output = self.forward_rgb(rgb)
        else:
            rgb_output = None
        
        return {
            'audio': audio_output,
            'rgb': rgb_output
        }
    
    def freeze_rgb(self):
        """Freeze RGB teacher parameters"""
        for param in self.rgb_encoder.parameters():
            param.requires_grad = False
        for param in self.rgb_bin_predictor.parameters():
            param.requires_grad = False
        for param in self.rgb_decoder.parameters():
            param.requires_grad = False
        print("RGB teacher frozen")
    
    def get_parameters_count(self):
        """Count parameters"""
        rgb_params = (
            sum(p.numel() for p in self.rgb_encoder.parameters()) +
            sum(p.numel() for p in self.rgb_bin_predictor.parameters()) +
            sum(p.numel() for p in self.rgb_decoder.parameters())
        )
        
        audio_params = (
            sum(p.numel() for p in self.audio_encoder.parameters()) +
            sum(p.numel() for p in self.audio_bin_predictor.parameters()) +
            sum(p.numel() for p in self.audio_decoder.parameters())
        )
        
        residual_params = sum(p.numel() for p in self.residual_head.parameters())
        
        return {
            'rgb_teacher': rgb_params,
            'audio_student': audio_params,
            'residual': residual_params,
            'total': rgb_params + audio_params + residual_params
        }


def create_adabins_distillation_model(
    n_bins=128,
    base_channels=64,
    output_size=256,
    max_depth=30.0,
    use_pretrained_rgb=False,
    gpu_ids=[]
):
    """
    Factory function to create AdaBins Distillation model
    
    Args:
        n_bins: Number of adaptive bins
        base_channels: Base channel count
        output_size: Output spatial resolution
        max_depth: Maximum depth value
        use_pretrained_rgb: Load pre-trained RGB encoder
        gpu_ids: GPU IDs for multi-GPU training
    
    Returns:
        model: Initialized model
    """
    model = AdaBinsDistillationModel(
        n_bins=n_bins,
        base_channels=base_channels,
        output_size=output_size,
        max_depth=max_depth,
        use_pretrained_rgb=use_pretrained_rgb
    )
    
    # Move to GPU if available
    if len(gpu_ids) > 0 and torch.cuda.is_available():
        model = model.to(f'cuda:{gpu_ids[0]}')
        if len(gpu_ids) > 1:
            model = nn.DataParallel(model, device_ids=gpu_ids)
    
    return model


# ==========================================
# Test Code
# ==========================================
if __name__ == '__main__':
    print("Testing AdaBins Distillation Model")
    print("=" * 60)
    
    model = AdaBinsDistillationModel(n_bins=128, base_channels=64, output_size=256)
    
    # Print parameter counts
    params = model.get_parameters_count()
    print(f"\nModel Parameters:")
    print(f"  RGB Teacher:   {params['rgb_teacher']:,}")
    print(f"  Audio Student: {params['audio_student']:,}")
    print(f"  Residual:      {params['residual']:,}")
    print(f"  Total:         {params['total']:,}")
    
    # Test training mode (with RGB)
    print(f"\n=== Training Mode (RGB + Audio) ===")
    batch_size = 4
    audio_input = torch.randn(batch_size, 2, 256, 256)
    rgb_input = torch.randn(batch_size, 3, 256, 256)
    
    output = model(audio_input, rgb_input, mode='train')
    
    print(f"Audio predictions:")
    print(f"  Bin centers:  {output['audio']['bin_centers'].shape}")
    print(f"  Base depth:   {output['audio']['base_depth'].shape}")
    print(f"  Residual:     {output['audio']['residual'].shape}")
    print(f"  Final depth:  {output['audio']['final_depth'].shape}")
    
    if output['rgb'] is not None:
        print(f"\nRGB predictions (teacher):")
        print(f"  Final depth:  {output['rgb']['final_depth'].shape}")
    
    # Test inference mode (Audio only)
    print(f"\n=== Inference Mode (Audio Only) ===")
    output_inference = model(audio_input, rgb=None, mode='inference')
    
    print(f"Audio predictions:")
    print(f"  Final depth:  {output_inference['audio']['final_depth'].shape}")
    print(f"  RGB output:   {output_inference['rgb']}")
    
    print("\n✅ Model test passed!")

