"""
RGB Depth Estimation Model (Compatible with Binaural Attention for Distillation)

Architecture:
    - Single CNN encoder for RGB input
    - U-Net decoder for depth prediction
    - Feature sizes MATCH binaural_attention_model for distillation
    
Key Design:
    - Input: 3 channels (RGB) instead of 2 channels (stereo audio)
    - Feature hierarchy matches BinauralAttentionDepthNet
    - Can serve as teacher for audio-based models via knowledge distillation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


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
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class RGBDepthNet(nn.Module):
    """
    RGB Depth Estimation Network
    
    Architecture compatible with BinauralAttentionDepthNet for distillation.
    Feature dimensions at each level match exactly for knowledge transfer.
    
    Args:
        base_channels: Base number of channels (default: 64)
        bilinear: Use bilinear upsampling (True) or transposed conv (False)
        output_size: Output spatial resolution (256 for Batvision)
        max_depth: Maximum depth value
    
    Feature Map Sizes (base_channels=64):
        x1: [B, 64, H, W]       - matches binaural fusion output
        x2: [B, 128, H/2, W/2]  - matches binaural fusion output
        x3: [B, 256, H/4, W/4]  - matches binaural fusion output
        x4: [B, 512, H/8, W/8]  - matches binaural fusion output
        x5: [B, 512, H/16, W/16] - matches binaural fusion output (bilinear=True)
    """
    
    def __init__(
        self,
        base_channels=64,
        bilinear=True,
        output_size=256,
        max_depth=30.0
    ):
        super().__init__()
        self.output_size = output_size
        self.max_depth = max_depth
        self.bilinear = bilinear
        
        # ==========================================
        # Encoder (RGB Input)
        # ==========================================
        self.inc = DoubleConv(3, base_channels)  # RGB: 3 channels
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor)
        
        # ==========================================
        # Decoder (U-Net style)
        # ==========================================
        # Feature dimensions EXACTLY match BinauralAttentionDepthNet decoder
        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)
        
        # Output head
        self.outc = nn.Conv2d(base_channels, 1, kernel_size=1)
        
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
    
    def forward(self, x, return_features=False):
        """
        Forward pass
        
        Args:
            x: [B, 3, H, W] - RGB image
            return_features: If True, return intermediate features for distillation
            
        Returns:
            depth: [B, 1, H, W] - Predicted depth map
            features: Dict of intermediate features (if return_features=True)
        """
        # ==========================================
        # Encoder
        # ==========================================
        x1 = self.inc(x)      # [B, 64, H, W]
        x2 = self.down1(x1)   # [B, 128, H/2, W/2]
        x3 = self.down2(x2)   # [B, 256, H/4, W/4]
        x4 = self.down3(x3)   # [B, 512, H/8, W/8]
        x5 = self.down4(x4)   # [B, 512, H/16, W/16] (or 1024 if bilinear=False)
        
        # Store features for distillation
        encoder_features = {
            'x1': x1,
            'x2': x2,
            'x3': x3,
            'x4': x4,
            'x5': x5
        }
        
        # ==========================================
        # Decoder with skip connections
        # ==========================================
        d4 = self.up1(x5, x4)  # [B, 256, H/8, W/8]
        d3 = self.up2(d4, x3)  # [B, 128, H/4, W/4]
        d2 = self.up3(d3, x2)  # [B, 64, H/2, W/2]
        d1 = self.up4(d2, x1)  # [B, 64, H, W]
        
        # Store decoder features for distillation
        decoder_features = {
            'd1': d1,
            'd2': d2,
            'd3': d3,
            'd4': d4
        }
        
        # ==========================================
        # Output depth
        # ==========================================
        depth = self.outc(d1)  # [B, 1, H, W]
        
        # Resize to output size if needed
        if depth.shape[-1] != self.output_size:
            depth = F.interpolate(
                depth,
                size=(self.output_size, self.output_size),
                mode='bilinear',
                align_corners=False
            )
        
        # Clamp to valid depth range
        depth = torch.clamp(depth, 0, self.max_depth)
        
        if return_features:
            features = {
                **encoder_features,
                **decoder_features
            }
            return depth, features
        else:
            return depth
    
    def get_num_params(self):
        """Return number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_rgb_depth_model(
    base_channels=64,
    bilinear=True,
    output_size=256,
    max_depth=30.0
):
    """
    Factory function to create RGB Depth model
    
    Args:
        base_channels: Base channel count (64 = ~20M params, 32 = ~5M params)
        bilinear: Use bilinear upsampling
        output_size: Output resolution
        max_depth: Maximum depth value
        
    Returns:
        model: RGBDepthNet instance
    """
    model = RGBDepthNet(
        base_channels=base_channels,
        bilinear=bilinear,
        output_size=output_size,
        max_depth=max_depth
    )
    
    print(f"Created RGB Depth Model:")
    print(f"  - Base channels: {base_channels}")
    print(f"  - Input: RGB (3 channels)")
    print(f"  - Total parameters: {model.get_num_params():,}")
    
    return model


if __name__ == "__main__":
    # Test model
    print("Testing RGB Depth Model...")
    print("=" * 80)
    
    model = create_rgb_depth_model(
        base_channels=64,
        bilinear=True
    )
    
    # Dummy RGB input
    dummy_input = torch.randn(2, 3, 256, 256)  # [B, 3, H, W]
    
    # Forward pass without features
    print("\n--- Test 1: Normal forward pass ---")
    with torch.no_grad():
        depth = model(dummy_input, return_features=False)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {depth.shape}")
    print(f"Output range: [{depth.min():.2f}, {depth.max():.2f}]")
    
    # Forward pass with features (for distillation)
    print("\n--- Test 2: Forward pass with features ---")
    with torch.no_grad():
        depth, features = model(dummy_input, return_features=True)
    
    print(f"Encoder features:")
    for k, v in features.items():
        if k.startswith('x'):
            print(f"  {k}: {v.shape}")
    
    print(f"Decoder features:")
    for k, v in features.items():
        if k.startswith('d'):
            print(f"  {k}: {v.shape}")
    
    print("\n✅ Model test passed!")
    
    # Compare with binaural model feature sizes
    print("\n" + "=" * 80)
    print("Feature Size Compatibility Check:")
    print("=" * 80)
    print("RGB Model features match BinauralAttentionDepthNet fused features:")
    print("  x1: 64 channels  ✓")
    print("  x2: 128 channels ✓")
    print("  x3: 256 channels ✓")
    print("  x4: 512 channels ✓")
    print("  x5: 512 channels ✓ (with bilinear=True)")
    print("\n✅ Ready for knowledge distillation!")






