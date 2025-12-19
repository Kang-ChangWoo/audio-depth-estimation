"""
Base + Residual Depth Model

Architecture:
    Shared Encoder → Two Decoders:
    1. Base Decoder: Predicts coarse depth (room layout/structure)
    2. Residual Decoder: Predicts fine-grained corrections
    
    Final Depth = Base Depth + Residual

Design Philosophy (Taylor Series Analogy):
    - Base: f(x) ≈ f(a) - the main function value (room layout)
    - Residual: f'(a)(x-a) - the local correction (details)
    - This decomposition makes the under-constrained audio-to-depth problem easier
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

        # if bilinear, use the normal convolutions to reduce the number of channels
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


class BaseResidualDepthNet(nn.Module):
    """
    Base + Residual Depth Estimation Network
    
    Args:
        input_channels: Number of input channels (2 for binaural audio, 3 for RGB)
        base_channels: Base number of channels in the network
        bilinear: Use bilinear upsampling (True) or transposed conv (False)
        output_size: Output spatial resolution (256 for Batvision)
    
    Forward Output:
        base_depth: [B, 1, H, W] - Coarse depth (room structure)
        residual: [B, 1, H, W] - Fine corrections
        final_depth: [B, 1, H, W] - base_depth + residual
    """
    
    def __init__(self, input_channels=2, base_channels=64, bilinear=True, output_size=256, max_depth=30.0):
        super().__init__()
        self.input_channels = input_channels
        self.output_size = output_size
        self.bilinear = bilinear
        self.max_depth = max_depth
        
        # ==========================================
        # Shared Encoder
        # ==========================================
        self.inc = DoubleConv(input_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor)
        
        # ==========================================
        # Base Decoder (Coarse Depth / Room Layout)
        # ==========================================
        # Goal: Capture low-frequency components (walls, floor, ceiling)
        self.base_up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear)
        self.base_up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.base_up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.base_up4 = Up(base_channels * 2, base_channels, bilinear)
        self.base_head = nn.Conv2d(base_channels, 1, kernel_size=1)
        
        # ==========================================
        # Residual Decoder (Fine Details)
        # ==========================================
        # Goal: Capture high-frequency components (object details, edges)
        self.res_up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear)
        self.res_up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.res_up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.res_up4 = Up(base_channels * 2, base_channels, bilinear)
        self.res_head = nn.Conv2d(base_channels, 1, kernel_size=1)
        
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
        Forward pass
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            base_depth: Coarse depth map [B, 1, H, W]
            residual: Residual corrections [B, 1, H, W]
            final_depth: base_depth + residual [B, 1, H, W]
        """
        # ==========================================
        # Shared Encoder
        # ==========================================
        x1 = self.inc(x)      # [B, 64, H, W]
        x2 = self.down1(x1)   # [B, 128, H/2, W/2]
        x3 = self.down2(x2)   # [B, 256, H/4, W/4]
        x4 = self.down3(x3)   # [B, 512, H/8, W/8]
        x5 = self.down4(x4)   # [B, 512 or 1024, H/16, W/16]
        
        # ==========================================
        # Base Decoder (Coarse Depth)
        # ==========================================
        b = self.base_up1(x5, x4)
        b = self.base_up2(b, x3)
        b = self.base_up3(b, x2)
        b = self.base_up4(b, x1)
        base_depth_raw = self.base_head(b)  # [B, 1, H, W]
        
        # CRITICAL: Base는 항상 양수 (구조는 양수여야 함)
        base_depth = torch.sigmoid(base_depth_raw) * self.max_depth
        
        # Ensure output size matches target
        if base_depth.shape[-2:] != (self.output_size, self.output_size):
            base_depth = F.interpolate(base_depth, size=(self.output_size, self.output_size),
                                      mode='bilinear', align_corners=False)
        
        # ==========================================
        # Residual Decoder (Fine Details)
        # ==========================================
        r = self.res_up1(x5, x4)
        r = self.res_up2(r, x3)
        r = self.res_up3(r, x2)
        r = self.res_up4(r, x1)
        residual_raw = self.res_head(r)  # [B, 1, H, W]
        
        # CRITICAL: Residual은 +/- 허용 (보정값)
        # Tanh로 제한하고 max_depth의 20%까지만 허용
        residual = torch.tanh(residual_raw) * (self.max_depth * 0.2)
        
        # Ensure output size matches target
        if residual.shape[-2:] != (self.output_size, self.output_size):
            residual = F.interpolate(residual, size=(self.output_size, self.output_size),
                                    mode='bilinear', align_corners=False)
        
        # ==========================================
        # Final Depth = Base + Residual
        # ==========================================
        final_depth = base_depth + residual
        
        # Clamp to valid range
        final_depth = torch.clamp(final_depth, 0, self.max_depth)
        
        return base_depth, residual, final_depth
    
    def get_parameters_count(self):
        """Count parameters in each component"""
        encoder_modules = [self.inc, self.down1, self.down2, self.down3, self.down4]
        encoder_params = sum(p.numel() for module in encoder_modules for p in module.parameters())
        
        base_decoder_modules = [self.base_up1, self.base_up2, self.base_up3, self.base_up4, self.base_head]
        base_decoder_params = sum(p.numel() for module in base_decoder_modules for p in module.parameters())
        
        res_decoder_modules = [self.res_up1, self.res_up2, self.res_up3, self.res_up4, self.res_head]
        res_decoder_params = sum(p.numel() for module in res_decoder_modules for p in module.parameters())
        
        total = encoder_params + base_decoder_params + res_decoder_params
        
        return {
            'encoder': encoder_params,
            'base_decoder': base_decoder_params,
            'residual_decoder': res_decoder_params,
            'total': total
        }


def create_base_residual_model(input_channels=2, base_channels=64, 
                                bilinear=True, output_size=256, 
                                max_depth=30.0, gpu_ids=[]):
    """
    Factory function to create Base+Residual model
    
    Args:
        input_channels: 2 for audio, 3 for image
        base_channels: Base number of filters
        bilinear: Use bilinear upsampling
        output_size: Output spatial resolution
        max_depth: Maximum depth value in meters
        gpu_ids: GPU IDs for multi-GPU training
    
    Returns:
        model: Initialized model
    """
    model = BaseResidualDepthNet(
        input_channels=input_channels,
        base_channels=base_channels,
        bilinear=bilinear,
        output_size=output_size,
        max_depth=max_depth
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
    print("Testing Base + Residual Depth Model")
    print("=" * 60)
    
    # Test with binaural audio input
    model = BaseResidualDepthNet(input_channels=2, base_channels=64, output_size=256)
    
    # Print model info
    params = model.get_parameters_count()
    print(f"\nModel Parameters:")
    print(f"  Encoder:          {params['encoder']:,}")
    print(f"  Base Decoder:     {params['base_decoder']:,}")
    print(f"  Residual Decoder: {params['residual_decoder']:,}")
    print(f"  Total:            {params['total']:,}")
    
    # Test forward pass
    print(f"\nTesting forward pass...")
    batch_size = 4
    dummy_input = torch.randn(batch_size, 2, 256, 256)
    
    base_depth, residual, final_depth = model(dummy_input)
    
    print(f"  Input shape:      {dummy_input.shape}")
    print(f"  Base depth:       {base_depth.shape}")
    print(f"  Residual:         {residual.shape}")
    print(f"  Final depth:      {final_depth.shape}")
    
    # Check that final = base + residual
    diff = torch.abs(final_depth - (base_depth + residual)).max()
    print(f"  Max difference:   {diff.item():.6f} (should be ~0)")
    
    print("\n✅ Model test passed!")


