"""
Coarse Depth Classification Model

Predicts depth as a classification problem over N bins.
Each pixel is classified into one of N discrete depth bins.

Model outputs:
- logits: [B, N_bins, H, W] - class logits for each pixel
- depth: [B, 1, H, W] - expected depth (soft prediction)

Supports multiple backbone architectures:
- UNet-style encoder-decoder
- Spline-based (adapted from SplineDepthV3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
from typing import Tuple, Dict, List, Optional


# ============================================================================
# UNet Backbone for Classification
# ============================================================================

class DoubleConv(nn.Module):
    """Double convolution block."""
    
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv."""
    
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )
    
    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv."""
    
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch, in_ch // 2)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle size mismatch
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class CoarseDepthUNet(nn.Module):
    """
    UNet-based model for coarse depth classification.
    
    Args:
        input_channels: Number of input channels (2 for binaural audio)
        n_bins: Number of depth bins for classification
        base_channels: Base number of channels
        bilinear: Use bilinear upsampling
    """
    
    def __init__(
        self,
        input_channels: int = 2,
        n_bins: int = 128,
        base_channels: int = 64,
        bilinear: bool = True,
        output_size: int = 256,
    ):
        super().__init__()
        self.n_bins = n_bins
        self.output_size = output_size
        
        factor = 2 if bilinear else 1
        
        # Encoder
        self.inc = DoubleConv(input_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor)
        
        # Decoder
        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)
        
        # Output head - produces N_bins channels for classification
        self.outc = nn.Conv2d(base_channels, n_bins, kernel_size=1)
        
        # Bin centers (will be set externally)
        self.register_buffer('bin_centers', torch.linspace(0, 1, n_bins))
    
    def set_bin_centers(self, bin_centers: torch.Tensor):
        """Set bin centers for depth reconstruction."""
        self.bin_centers = bin_centers.to(self.bin_centers.device)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            logits: [B, N_bins, H, W] - class logits
            depth: [B, 1, H, W] - expected depth
        """
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output logits
        logits = self.outc(x)  # [B, N_bins, H, W]
        
        # Resize to output size if needed
        if logits.shape[-1] != self.output_size:
            logits = F.interpolate(logits, size=(self.output_size, self.output_size),
                                   mode='bilinear', align_corners=False)
        
        # Compute expected depth from softmax probabilities
        probs = F.softmax(logits, dim=1)  # [B, N_bins, H, W]
        bin_centers = self.bin_centers.view(1, -1, 1, 1)  # [1, N_bins, 1, 1]
        depth = (probs * bin_centers).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        
        return logits, depth
    
    def predict_depth(self, x: torch.Tensor, mode: str = 'soft') -> torch.Tensor:
        """
        Predict depth map.
        
        Args:
            x: Input audio spectrogram
            mode: 'soft' for expected value, 'hard' for argmax
        
        Returns:
            depth: [B, 1, H, W]
        """
        logits, soft_depth = self.forward(x)
        
        if mode == 'soft':
            return soft_depth
        else:
            # Hard prediction (argmax)
            bin_indices = logits.argmax(dim=1, keepdim=True)  # [B, 1, H, W]
            depth = self.bin_centers[bin_indices.squeeze(1)]  # [B, H, W]
            return depth.unsqueeze(1)
    
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Lightweight Classification Head
# ============================================================================

class CoarseDepthLite(nn.Module):
    """
    Lightweight coarse depth model using simple encoder-decoder.
    Faster training, good for initial experiments.
    """
    
    def __init__(
        self,
        input_channels: int = 2,
        n_bins: int = 128,
        base_channels: int = 48,
        output_size: int = 256,
    ):
        super().__init__()
        self.n_bins = n_bins
        self.output_size = output_size
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, 4, 2, 1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(base_channels * 8, base_channels * 8, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 8, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(base_channels, base_channels, 4, 2, 1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        
        # Classification head
        self.head = nn.Conv2d(base_channels, n_bins, 3, 1, 1)
        
        self.register_buffer('bin_centers', torch.linspace(0, 1, n_bins))
    
    def set_bin_centers(self, bin_centers: torch.Tensor):
        self.bin_centers = bin_centers.to(self.bin_centers.device)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.encoder(x)
        feat = self.decoder(feat)
        
        logits = self.head(feat)
        
        if logits.shape[-1] != self.output_size:
            logits = F.interpolate(logits, size=(self.output_size, self.output_size),
                                   mode='bilinear', align_corners=False)
        
        probs = F.softmax(logits, dim=1)
        bin_centers = self.bin_centers.view(1, -1, 1, 1)
        depth = (probs * bin_centers).sum(dim=1, keepdim=True)
        
        return logits, depth
    
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Loss Functions for Classification
# ============================================================================

class OrdinalRegressionLoss(nn.Module):
    """
    Ordinal regression loss for depth classification.
    Treats depth bins as ordered classes.
    """
    
    def __init__(self, n_bins: int, weight: float = 1.0):
        super().__init__()
        self.n_bins = n_bins
        self.weight = weight
    
    def forward(self, logits: torch.Tensor, target_bins: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, N_bins, H, W]
            target_bins: [B, H, W] - target bin indices
        """
        B, N, H, W = logits.shape
        
        # Create ordinal labels: for each pixel, bins <= target should be 1
        target_bins = target_bins.unsqueeze(1)  # [B, 1, H, W]
        bin_indices = torch.arange(N, device=logits.device).view(1, N, 1, 1)
        ordinal_labels = (bin_indices <= target_bins).float()  # [B, N, H, W]
        
        # Binary cross entropy for each bin
        loss = F.binary_cross_entropy_with_logits(logits, ordinal_labels)
        
        return self.weight * loss


class SoftCrossEntropyLoss(nn.Module):
    """
    Soft cross entropy with label smoothing around target bin.
    """
    
    def __init__(self, n_bins: int, sigma: float = 2.0, weight: float = 1.0):
        super().__init__()
        self.n_bins = n_bins
        self.sigma = sigma
        self.weight = weight
    
    def forward(self, logits: torch.Tensor, target_bins: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, N_bins, H, W]
            target_bins: [B, H, W] - target bin indices
        """
        B, N, H, W = logits.shape
        
        # Create soft labels (Gaussian around target bin)
        target_bins = target_bins.unsqueeze(1).float()  # [B, 1, H, W]
        bin_indices = torch.arange(N, device=logits.device, dtype=torch.float32).view(1, N, 1, 1)
        
        # Gaussian distribution centered at target
        soft_labels = torch.exp(-0.5 * ((bin_indices - target_bins) / self.sigma) ** 2)
        soft_labels = soft_labels / (soft_labels.sum(dim=1, keepdim=True) + 1e-8)
        
        # Cross entropy with soft labels
        log_probs = F.log_softmax(logits, dim=1)
        loss = -(soft_labels * log_probs).sum(dim=1).mean()
        
        return self.weight * loss


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance.
    """
    
    def __init__(self, gamma: float = 2.0, weight: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
    
    def forward(self, logits: torch.Tensor, target_bins: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, N_bins, H, W]
            target_bins: [B, H, W] - target bin indices (long)
        """
        B, N, H, W = logits.shape
        
        # Reshape for cross entropy
        logits = logits.permute(0, 2, 3, 1).reshape(-1, N)  # [B*H*W, N]
        targets = target_bins.reshape(-1)  # [B*H*W]
        
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        return self.weight * focal_loss.mean()


# ============================================================================
# Combined Loss for Coarse Depth
# ============================================================================

class CoarseDepthLoss(nn.Module):
    """
    Combined loss for coarse depth classification.
    
    Combines:
    - Classification loss (CE or focal)
    - Depth regression loss (on soft predictions)
    - Optional ordinal loss
    """
    
    def __init__(
        self,
        n_bins: int = 128,
        ce_weight: float = 1.0,
        regression_weight: float = 0.5,
        use_focal: bool = False,
        focal_gamma: float = 2.0,
        use_soft_ce: bool = True,
        soft_ce_sigma: float = 2.0,
    ):
        super().__init__()
        
        self.ce_weight = ce_weight
        self.regression_weight = regression_weight
        
        if use_focal:
            self.ce_loss = FocalLoss(gamma=focal_gamma)
        elif use_soft_ce:
            self.ce_loss = SoftCrossEntropyLoss(n_bins, sigma=soft_ce_sigma)
        else:
            self.ce_loss = nn.CrossEntropyLoss()
        
        self.regression_loss = nn.L1Loss()
        self.use_soft_ce = use_soft_ce
    
    def forward(
        self,
        logits: torch.Tensor,
        pred_depth: torch.Tensor,
        target_bins: torch.Tensor,
        target_depth: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            logits: [B, N_bins, H, W]
            pred_depth: [B, 1, H, W]
            target_bins: [B, H, W] or [B, 1, H, W]
            target_depth: [B, 1, H, W]
            valid_mask: [B, 1, H, W] or None
        """
        # Handle target_bins shape
        if target_bins.dim() == 4:
            target_bins = target_bins.squeeze(1)
        
        # Classification loss
        if self.use_soft_ce:
            ce_loss = self.ce_loss(logits, target_bins)
        else:
            # Standard CE needs long targets
            ce_loss = self.ce_loss(logits, target_bins.long())
        
        # Regression loss
        if valid_mask is not None:
            reg_loss = self.regression_loss(
                pred_depth[valid_mask], 
                target_depth[valid_mask]
            )
        else:
            reg_loss = self.regression_loss(pred_depth, target_depth)
        
        total_loss = self.ce_weight * ce_loss + self.regression_weight * reg_loss
        
        return {
            'total': total_loss,
            'ce': ce_loss,
            'regression': reg_loss,
        }


# ============================================================================
# Model Factory
# ============================================================================

def init_weights(net, init_type='kaiming', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)


def init_net(net, init_type='kaiming', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert torch.cuda.is_available()
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, init_gain)
    return net


def define_coarse_depth_model(
    model_type: str = 'unet',
    input_channels: int = 2,
    n_bins: int = 128,
    base_channels: int = 64,
    output_size: int = 256,
    init_type: str = 'kaiming',
    gpu_ids: List[int] = [],
) -> nn.Module:
    """
    Factory function to create coarse depth model.
    
    Args:
        model_type: 'unet' or 'lite'
        input_channels: Number of input channels
        n_bins: Number of depth bins
        base_channels: Base channel count
        output_size: Output spatial size
        init_type: Weight initialization type
        gpu_ids: List of GPU IDs
    """
    if model_type == 'unet':
        net = CoarseDepthUNet(
            input_channels=input_channels,
            n_bins=n_bins,
            base_channels=base_channels,
            output_size=output_size,
        )
    elif model_type == 'lite':
        net = CoarseDepthLite(
            input_channels=input_channels,
            n_bins=n_bins,
            base_channels=base_channels,
            output_size=output_size,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    return init_net(net, init_type, gpu_ids=gpu_ids)


# ============================================================================
# Testing
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Testing Coarse Depth Models")
    print("=" * 60)
    
    batch_size = 2
    n_bins = 128
    input_size = 256
    
    # Test UNet
    print("\n1. Testing CoarseDepthUNet...")
    model = CoarseDepthUNet(input_channels=2, n_bins=n_bins, output_size=input_size)
    audio = torch.randn(batch_size, 2, input_size, input_size)
    
    logits, depth = model(audio)
    print(f"   Input: {audio.shape}")
    print(f"   Logits: {logits.shape}")
    print(f"   Depth: {depth.shape}")
    print(f"   Parameters: {model.get_num_params():,}")
    
    # Test Lite
    print("\n2. Testing CoarseDepthLite...")
    model_lite = CoarseDepthLite(input_channels=2, n_bins=n_bins, output_size=input_size)
    logits_lite, depth_lite = model_lite(audio)
    print(f"   Logits: {logits_lite.shape}")
    print(f"   Depth: {depth_lite.shape}")
    print(f"   Parameters: {model_lite.get_num_params():,}")
    
    # Test loss
    print("\n3. Testing CoarseDepthLoss...")
    loss_fn = CoarseDepthLoss(n_bins=n_bins, use_soft_ce=True)
    target_bins = torch.randint(0, n_bins, (batch_size, input_size, input_size))
    target_depth = torch.rand(batch_size, 1, input_size, input_size)
    
    losses = loss_fn(logits, depth, target_bins, target_depth)
    print(f"   Total: {losses['total']:.4f}")
    print(f"   CE: {losses['ce']:.4f}")
    print(f"   Regression: {losses['regression']:.4f}")
    
    print("\nAll tests passed!")


# ============================================================================
# Hybrid Model: Coarse Classification + Offset Regression
# ============================================================================

class CoarseWithOffsetModel(nn.Module):
    """
    Hybrid model combining:
    1. Coarse classification (small bins, e.g., 8) for rough depth estimation
    2. Offset regression that uses BOTH encoder features AND coarse prediction
    
    Final depth = coarse_depth + offset
    
    This design allows:
    - Classification to capture major depth regions
    - Regression to refine with continuous offset
    - Offset regressor has access to audio features for context
    """
    
    def __init__(
        self,
        input_channels: int = 2,
        n_bins: int = 8,  # Small number of bins (single digit)
        base_channels: int = 64,
        output_size: int = 256,
        bilinear: bool = True,
    ):
        super().__init__()
        self.n_bins = n_bins
        self.output_size = output_size
        
        factor = 2 if bilinear else 1
        
        # =====================
        # Shared Encoder
        # =====================
        self.inc = DoubleConv(input_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor)
        
        # =====================
        # Coarse Classification Decoder
        # =====================
        self.coarse_up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear)
        self.coarse_up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.coarse_up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.coarse_up4 = Up(base_channels * 2, base_channels, bilinear)
        self.coarse_head = nn.Conv2d(base_channels, n_bins, kernel_size=1)
        
        # =====================
        # Offset Regression Decoder
        # Takes encoder features + coarse depth (1 channel)
        # =====================
        # Additional input: coarse depth (1 channel) concatenated with encoder features
        self.offset_up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear)
        self.offset_up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.offset_up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.offset_up4 = Up(base_channels * 2, base_channels, bilinear)
        
        # Fusion layer: combines decoder output + coarse depth
        self.offset_fusion = nn.Sequential(
            nn.Conv2d(base_channels + 1, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels // 2, 3, padding=1),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(inplace=True),
        )
        self.offset_head = nn.Conv2d(base_channels // 2, 1, kernel_size=1)
        
        # Bin centers (will be set externally or default to linear)
        self.register_buffer('bin_centers', torch.linspace(0, 1, n_bins))
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def set_bin_centers(self, bin_centers: torch.Tensor):
        """Set bin centers for depth reconstruction."""
        self.bin_centers = bin_centers.to(self.bin_centers.device)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            logits: [B, N_bins, H, W] - coarse classification logits
            coarse_depth: [B, 1, H, W] - coarse depth from classification
            offset: [B, 1, H, W] - predicted offset
            final_depth: [B, 1, H, W] - coarse_depth + offset
        """
        # =====================
        # Shared Encoder
        # =====================
        x1 = self.inc(x)      # [B, 64, H, W]
        x2 = self.down1(x1)   # [B, 128, H/2, W/2]
        x3 = self.down2(x2)   # [B, 256, H/4, W/4]
        x4 = self.down3(x3)   # [B, 512, H/8, W/8]
        x5 = self.down4(x4)   # [B, 512, H/16, W/16]
        
        # =====================
        # Coarse Classification Branch
        # =====================
        c = self.coarse_up1(x5, x4)
        c = self.coarse_up2(c, x3)
        c = self.coarse_up3(c, x2)
        c = self.coarse_up4(c, x1)
        logits = self.coarse_head(c)  # [B, N_bins, H, W]
        
        # Resize logits if needed (check both height and width)
        if logits.shape[-2] != self.output_size or logits.shape[-1] != self.output_size:
            logits = F.interpolate(logits, size=(self.output_size, self.output_size),
                                   mode='bilinear', align_corners=False)
        
        # Compute coarse depth from softmax
        probs = F.softmax(logits, dim=1)
        bin_centers = self.bin_centers.view(1, -1, 1, 1)
        coarse_depth = (probs * bin_centers).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        
        # =====================
        # Offset Regression Branch
        # Uses encoder features + coarse depth
        # =====================
        o = self.offset_up1(x5, x4)
        o = self.offset_up2(o, x3)
        o = self.offset_up3(o, x2)
        o = self.offset_up4(o, x1)
        
        # Resize offset features if needed (check both height and width)
        if o.shape[-2] != self.output_size or o.shape[-1] != self.output_size:
            o = F.interpolate(o, size=(self.output_size, self.output_size),
                              mode='bilinear', align_corners=False)
        
        # Concatenate decoder features with coarse depth
        # This gives the offset regressor access to both audio features AND coarse prediction
        o_with_coarse = torch.cat([o, coarse_depth.detach()], dim=1)  # [B, 64+1, H, W]
        
        # Fuse and predict offset
        o_fused = self.offset_fusion(o_with_coarse)
        offset = self.offset_head(o_fused)  # [B, 1, H, W]
        
        # Final depth = coarse + offset
        final_depth = coarse_depth + offset
        
        return logits, coarse_depth, offset, final_depth
    
    def predict_depth(self, x: torch.Tensor) -> torch.Tensor:
        """Predict final depth map."""
        _, _, _, final_depth = self.forward(x)
        return final_depth
    
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_component_params(self) -> Dict:
        """Get parameter count for each component."""
        encoder_params = sum(p.numel() for p in [
            self.inc, self.down1, self.down2, self.down3, self.down4
        ] for p in p.parameters() if p.requires_grad)
        
        coarse_params = sum(p.numel() for p in [
            self.coarse_up1, self.coarse_up2, self.coarse_up3, self.coarse_up4, self.coarse_head
        ] for p in p.parameters() if p.requires_grad)
        
        offset_params = sum(p.numel() for p in [
            self.offset_up1, self.offset_up2, self.offset_up3, self.offset_up4,
            self.offset_fusion, self.offset_head
        ] for p in p.parameters() if p.requires_grad)
        
        return {
            'encoder': encoder_params,
            'coarse_decoder': coarse_params,
            'offset_decoder': offset_params,
            'total': self.get_num_params()
        }


class CoarseOffsetLoss(nn.Module):
    """
    Loss function for CoarseWithOffsetModel.
    
    Combines:
    1. Cross-entropy for coarse classification
    2. L1/L2 loss for final depth regression
    3. Optional: offset regularization (keep offset small)
    """
    
    def __init__(
        self,
        ce_weight: float = 1.0,
        regression_weight: float = 1.0,
        offset_reg_weight: float = 0.1,
        regression_loss: str = 'l1',  # 'l1' or 'l2'
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.regression_weight = regression_weight
        self.offset_reg_weight = offset_reg_weight
        
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.regression_loss_type = regression_loss
    
    def forward(
        self,
        logits: torch.Tensor,
        coarse_depth: torch.Tensor,
        offset: torch.Tensor,
        final_depth: torch.Tensor,
        target_depth: torch.Tensor,
        target_bins: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss.
        
        Args:
            logits: [B, N_bins, H, W] - classification logits
            coarse_depth: [B, 1, H, W] - coarse depth prediction
            offset: [B, 1, H, W] - predicted offset
            final_depth: [B, 1, H, W] - final depth (coarse + offset)
            target_depth: [B, 1, H, W] - ground truth depth
            target_bins: [B, H, W] - ground truth bin indices (long tensor)
        
        Returns:
            total_loss: Combined loss
            loss_dict: Individual loss components
        """
        # 1. Cross-entropy loss for coarse classification
        ce_loss = self.ce_loss(logits, target_bins)
        
        # 2. Regression loss for final depth
        if self.regression_loss_type == 'l1':
            reg_loss = F.l1_loss(final_depth, target_depth)
        else:
            reg_loss = F.mse_loss(final_depth, target_depth)
        
        # 3. Offset regularization (encourage small offsets)
        offset_reg = offset.abs().mean()
        
        # Combine
        total_loss = (
            self.ce_weight * ce_loss +
            self.regression_weight * reg_loss +
            self.offset_reg_weight * offset_reg
        )
        
        loss_dict = {
            'total': total_loss,
            'ce': ce_loss,
            'regression': reg_loss,
            'offset_reg': offset_reg,
            'coarse_l1': F.l1_loss(coarse_depth, target_depth),  # For monitoring
        }
        
        return total_loss, loss_dict


# ============================================================================
# Pure Regression Model: Coarse Regression + Offset Regression
# ============================================================================

class DualRegressionModel(nn.Module):
    """
    Pure regression model (no classification):
    1. Coarse regression: predicts rough depth directly
    2. Offset regression: refines using encoder features + coarse prediction
    
    Final depth = coarse_depth + offset
    
    This avoids classification issues and is simpler to train.
    """
    
    def __init__(
        self,
        input_channels: int = 2,
        base_channels: int = 64,
        output_size: int = 256,
        bilinear: bool = True,
    ):
        super().__init__()
        self.output_size = output_size
        
        factor = 2 if bilinear else 1
        
        # =====================
        # Shared Encoder
        # =====================
        self.inc = DoubleConv(input_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor)
        
        # =====================
        # Coarse Regression Decoder
        # =====================
        self.coarse_up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear)
        self.coarse_up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.coarse_up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.coarse_up4 = Up(base_channels * 2, base_channels, bilinear)
        self.coarse_head = nn.Conv2d(base_channels, 1, kernel_size=1)  # Direct regression
        
        # =====================
        # Offset Regression Decoder
        # Takes encoder features + coarse depth (1 channel)
        # =====================
        self.offset_up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear)
        self.offset_up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.offset_up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.offset_up4 = Up(base_channels * 2, base_channels, bilinear)
        
        # Fusion layer: combines decoder output + coarse depth
        self.offset_fusion = nn.Sequential(
            nn.Conv2d(base_channels + 1, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels // 2, 3, padding=1),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(inplace=True),
        )
        self.offset_head = nn.Conv2d(base_channels // 2, 1, kernel_size=1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            coarse_depth: [B, 1, H, W] - coarse depth from regression
            offset: [B, 1, H, W] - predicted offset
            final_depth: [B, 1, H, W] - coarse_depth + offset
        """
        # =====================
        # Shared Encoder
        # =====================
        x1 = self.inc(x)      # [B, 64, H, W]
        x2 = self.down1(x1)   # [B, 128, H/2, W/2]
        x3 = self.down2(x2)   # [B, 256, H/4, W/4]
        x4 = self.down3(x3)   # [B, 512, H/8, W/8]
        x5 = self.down4(x4)   # [B, 512, H/16, W/16]
        
        # =====================
        # Coarse Regression Branch
        # =====================
        c = self.coarse_up1(x5, x4)
        c = self.coarse_up2(c, x3)
        c = self.coarse_up3(c, x2)
        c = self.coarse_up4(c, x1)
        coarse_depth = self.coarse_head(c)  # [B, 1, H, W]
        
        # Resize if needed
        if coarse_depth.shape[-2] != self.output_size or coarse_depth.shape[-1] != self.output_size:
            coarse_depth = F.interpolate(coarse_depth, size=(self.output_size, self.output_size),
                                         mode='bilinear', align_corners=False)
        
        # =====================
        # Offset Regression Branch
        # Uses encoder features + coarse depth
        # =====================
        o = self.offset_up1(x5, x4)
        o = self.offset_up2(o, x3)
        o = self.offset_up3(o, x2)
        o = self.offset_up4(o, x1)
        
        # Resize if needed
        if o.shape[-2] != self.output_size or o.shape[-1] != self.output_size:
            o = F.interpolate(o, size=(self.output_size, self.output_size),
                              mode='bilinear', align_corners=False)
        
        # Concatenate decoder features with coarse depth (detached to prevent gradient flow)
        o_with_coarse = torch.cat([o, coarse_depth.detach()], dim=1)  # [B, 64+1, H, W]
        
        # Fuse and predict offset
        o_fused = self.offset_fusion(o_with_coarse)
        offset = self.offset_head(o_fused)  # [B, 1, H, W]
        
        # Final depth = coarse + offset
        final_depth = coarse_depth + offset
        
        return coarse_depth, offset, final_depth
    
    def predict_depth(self, x: torch.Tensor) -> torch.Tensor:
        """Predict final depth map."""
        _, _, final_depth = self.forward(x)
        return final_depth
    
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DualRegressionLoss(nn.Module):
    """
    Loss function for DualRegressionModel.
    
    Combines:
    1. L1 loss for coarse depth
    2. L1 loss for final depth
    3. Offset regularization (optional, keep offset small)
    """
    
    def __init__(
        self,
        coarse_weight: float = 1.0,
        final_weight: float = 1.0,
        offset_reg_weight: float = 0.01,
    ):
        super().__init__()
        self.coarse_weight = coarse_weight
        self.final_weight = final_weight
        self.offset_reg_weight = offset_reg_weight
    
    def forward(
        self,
        coarse_depth: torch.Tensor,
        offset: torch.Tensor,
        final_depth: torch.Tensor,
        target_depth: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss.
        """
        # Valid mask (depth > 0)
        valid_mask = target_depth > 0
        
        # 1. Coarse depth loss
        if valid_mask.any():
            coarse_loss = F.l1_loss(coarse_depth[valid_mask], target_depth[valid_mask])
            final_loss = F.l1_loss(final_depth[valid_mask], target_depth[valid_mask])
        else:
            coarse_loss = F.l1_loss(coarse_depth, target_depth)
            final_loss = F.l1_loss(final_depth, target_depth)
        
        # 2. Offset regularization
        offset_reg = offset.abs().mean()
        
        # Combine
        total_loss = (
            self.coarse_weight * coarse_loss +
            self.final_weight * final_loss +
            self.offset_reg_weight * offset_reg
        )
        
        loss_dict = {
            'total': total_loss,
            'coarse': coarse_loss,
            'final': final_loss,
            'offset_reg': offset_reg,
        }
        
        return total_loss, loss_dict

