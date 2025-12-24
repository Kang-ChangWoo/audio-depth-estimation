"""
Binaural Attention Depth Estimation Model

Architecture:
    1. Split left/right channels
    2. Process each channel through separate encoders
    3. Apply cross-attention between left and right features at multiple scales
    4. Fuse features and decode to depth map
    
Key Innovation:
    - Explicit binaural correspondence modeling through attention
    - Captures ITD (Inter-aural Time Difference) and ILD (Inter-aural Level Difference)
    - Multi-scale cross-attention for robust spatial cues
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


class BinauralCrossAttention(nn.Module):
    """
    Cross-attention between left and right channel features
    
    This module computes attention from one channel to another,
    allowing the network to explicitly model correspondence and
    binaural cues (ITD, ILD).
    """
    
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        
        # Query, Key, Value projections
        self.query = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.key = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        
        # Output projection
        self.out = nn.Conv2d(channels, channels, kernel_size=1)
        
        # Learnable scaling factor
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, left_feat, right_feat):
        """
        Args:
            left_feat: [B, C, H, W] - Left channel features
            right_feat: [B, C, H, W] - Right channel features
            
        Returns:
            left_attended: [B, C, H, W] - Left features attended to right
            right_attended: [B, C, H, W] - Right features attended to left
        """
        B, C, H, W = left_feat.shape
        
        # Left attends to Right
        Q_left = self.query(left_feat).view(B, -1, H * W)  # [B, C', HW]
        K_right = self.key(right_feat).view(B, -1, H * W)  # [B, C', HW]
        V_right = self.value(right_feat).view(B, C, H * W)  # [B, C, HW]
        
        # Compute attention weights
        attention_left = torch.softmax(
            torch.bmm(Q_left.transpose(1, 2), K_right) / (C ** 0.5),  # [B, HW, HW]
            dim=-1
        )
        
        # Apply attention to values
        attended_left = torch.bmm(V_right, attention_left.transpose(1, 2))  # [B, C, HW]
        attended_left = attended_left.view(B, C, H, W)
        attended_left = self.out(attended_left)
        
        # Residual connection with learnable weight
        left_out = left_feat + self.gamma * attended_left
        
        # Right attends to Left (symmetric)
        Q_right = self.query(right_feat).view(B, -1, H * W)
        K_left = self.key(left_feat).view(B, -1, H * W)
        V_left = self.value(left_feat).view(B, C, H * W)
        
        attention_right = torch.softmax(
            torch.bmm(Q_right.transpose(1, 2), K_left) / (C ** 0.5),
            dim=-1
        )
        
        attended_right = torch.bmm(V_left, attention_right.transpose(1, 2))
        attended_right = attended_right.view(B, C, H, W)
        attended_right = self.out(attended_right)
        
        right_out = right_feat + self.gamma * attended_right
        
        return left_out, right_out


class BinauralEncoder(nn.Module):
    """
    Encoder for single channel (left or right)
    Extracts hierarchical features
    """
    
    def __init__(self, base_channels=64, bilinear=True):
        super().__init__()
        self.inc = DoubleConv(1, base_channels)  # Single channel input
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor)
        
    def forward(self, x):
        """Returns multi-scale features"""
        x1 = self.inc(x)      # 64 channels
        x2 = self.down1(x1)   # 128 channels
        x3 = self.down2(x2)   # 256 channels
        x4 = self.down3(x3)   # 512 channels
        x5 = self.down4(x4)   # 512 channels (bottleneck)
        return {'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5}


class BinauralAttentionDepthNet(nn.Module):
    """
    Binaural Attention Depth Estimation Network
    
    Architecture:
        1. Separate encoders for left/right channels
        2. Multi-scale cross-attention between channels
        3. Feature fusion
        4. U-Net decoder to depth map
    
    Args:
        base_channels: Base number of channels in the network
        bilinear: Use bilinear upsampling (True) or transposed conv (False)
        output_size: Output spatial resolution (256 for Batvision)
        max_depth: Maximum depth value
        attention_levels: Which encoder levels to apply attention [1,2,3,4,5]
    """
    
    def __init__(
        self,
        base_channels=64,
        bilinear=True,
        output_size=256,
        max_depth=30.0,
        attention_levels=[2, 3, 4, 5]  # Apply attention at these levels
    ):
        super().__init__()
        self.output_size = output_size
        self.max_depth = max_depth
        self.bilinear = bilinear
        self.attention_levels = attention_levels
        
        # ==========================================
        # Separate Encoders for Left and Right
        # ==========================================
        self.left_encoder = BinauralEncoder(base_channels, bilinear)
        self.right_encoder = BinauralEncoder(base_channels, bilinear)
        
        # ==========================================
        # Cross-Attention Modules at Multiple Scales
        # ==========================================
        self.attention_modules = nn.ModuleDict()
        channel_map = {
            1: base_channels,
            2: base_channels * 2,
            3: base_channels * 4,
            4: base_channels * 8,
            5: base_channels * 8 if bilinear else base_channels * 16
        }
        
        for level in attention_levels:
            self.attention_modules[f'attn_{level}'] = BinauralCrossAttention(
                channels=channel_map[level],
                reduction=8
            )
        
        # ==========================================
        # Fusion layers (combine left and right)
        # ==========================================
        self.fusion_layers = nn.ModuleDict()
        for level in [1, 2, 3, 4, 5]:
            ch = channel_map[level]
            # Fuse by concatenation + 1x1 conv
            self.fusion_layers[f'fusion_{level}'] = nn.Sequential(
                nn.Conv2d(ch * 2, ch, kernel_size=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True)
            )
        
        # ==========================================
        # Decoder (U-Net style)
        # ==========================================
        factor = 2 if bilinear else 1
        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)
        
        # Output head
        # FIX: Add Sigmoid activation to ensure positive depth predictions
        # Output range: [0, 1] which will be scaled to [0, max_depth]
        self.outc = nn.Sequential(
            nn.Conv2d(base_channels, 1, kernel_size=1),
            nn.Sigmoid()  # Ensures output is in [0, 1] range
        )
        
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
    
    def forward(self, x):
        """
        Args:
            x: [B, 2, H, W] - Binaural audio spectrogram
            
        Returns:
            depth: [B, 1, H, W] - Predicted depth map
            attention_maps: Dict of attention visualizations (optional)
        """
        # Split into left and right channels
        left = x[:, 0:1, :, :]   # [B, 1, H, W]
        right = x[:, 1:2, :, :]  # [B, 1, H, W]
        
        # Encode separately
        left_feats = self.left_encoder(left)
        right_feats = self.right_encoder(right)
        
        # Apply cross-attention at specified levels
        attention_maps = {}
        for level in [1, 2, 3, 4, 5]:
            left_feat = left_feats[f'x{level}']
            right_feat = right_feats[f'x{level}']
            
            if level in self.attention_levels:
                # Apply attention
                left_feat, right_feat = self.attention_modules[f'attn_{level}'](
                    left_feat, right_feat
                )
                # Store for visualization (optional)
                attention_maps[f'level_{level}'] = (left_feat, right_feat)
            
            # Fuse left and right
            fused = self.fusion_layers[f'fusion_{level}'](
                torch.cat([left_feat, right_feat], dim=1)
            )
            left_feats[f'x{level}'] = fused
        
        # Decoder with skip connections
        x = self.up1(left_feats['x5'], left_feats['x4'])
        x = self.up2(x, left_feats['x3'])
        x = self.up3(x, left_feats['x2'])
        x = self.up4(x, left_feats['x1'])
        
        # Output depth
        # FIX: outc now outputs [0, 1] via Sigmoid, scale to [0, max_depth]
        depth_normalized = self.outc(x)  # [B, 1, H, W] in range [0, 1]
        depth = depth_normalized * self.max_depth  # Scale to [0, max_depth]
        
        # Resize to output size if needed
        if depth.shape[-1] != self.output_size:
            depth = F.interpolate(
                depth,
                size=(self.output_size, self.output_size),
                mode='bilinear',
                align_corners=False
            )
        
        # Clamp to valid depth range (safety, should already be in range)
        depth = torch.clamp(depth, 0, self.max_depth)
        
        return depth
    
    def get_num_params(self):
        """Return number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_binaural_attention_model(
    base_channels=64,
    bilinear=True,
    output_size=256,
    max_depth=30.0,
    attention_levels=[2, 3, 4, 5]
):
    """
    Factory function to create Binaural Attention model
    
    Args:
        base_channels: Base channel count (64 = ~40M params, 32 = ~10M params)
        bilinear: Use bilinear upsampling
        output_size: Output resolution
        max_depth: Maximum depth value
        attention_levels: Which encoder levels to apply attention
        
    Returns:
        model: BinauralAttentionDepthNet instance
    """
    model = BinauralAttentionDepthNet(
        base_channels=base_channels,
        bilinear=bilinear,
        output_size=output_size,
        max_depth=max_depth,
        attention_levels=attention_levels
    )
    
    print(f"Created Binaural Attention Model:")
    print(f"  - Base channels: {base_channels}")
    print(f"  - Attention levels: {attention_levels}")
    print(f"  - Total parameters: {model.get_num_params():,}")
    
    return model


if __name__ == "__main__":
    # Test model
    print("Testing Binaural Attention Model...")
    
    model = create_binaural_attention_model(
        base_channels=64,
        attention_levels=[2, 3, 4, 5]
    )
    
    # Dummy input
    dummy_input = torch.randn(2, 2, 256, 256)  # [B, 2, H, W]
    
    # Forward pass
    with torch.no_grad():
        depth = model(dummy_input)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {depth.shape}")
    print(f"Output range: [{depth.min():.2f}, {depth.max():.2f}]")
    print("\n✅ Model test passed!")

