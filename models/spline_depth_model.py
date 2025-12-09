"""
SplineGS-inspired Audio-to-Depth Model

This model predicts depth maps from binaural audio using a low-rank x,y spline decomposition:
    D(x,y) = Σ_r u_r(x) * v_r(y)

where u_r and v_r are cubic Hermite splines parameterized by learned control points.

Key concepts from SplineGS:
1. MAS (Motion-Adaptive Spline): Use cubic Hermite splines for smooth representation
2. MACP (Motion-Adaptive Control Point Pruning): Adaptive complexity via LS-based pruning
3. Low-rank factorization: Decompose 2D depth into product of 1D splines

References:
- SplineGS: https://cvpr.thecvf.com/virtual/2024/poster/SplineGS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
import math
from typing import Tuple, Optional, List


# ============================================================================
# Spline Utilities
# ============================================================================

class CubicHermiteSpline(nn.Module):
    """
    Evaluates cubic Hermite spline from control points.
    
    Given control points P = {p_0, p_1, ..., p_N-1} and positions t ∈ [0, 1],
    compute spline values using Catmull-Rom tangent estimation:
        m_k = (p_{k+1} - p_{k-1}) / 2
    
    The Hermite basis functions are:
        h00(τ) = 2τ³ - 3τ² + 1
        h10(τ) = τ³ - 2τ² + τ
        h01(τ) = -2τ³ + 3τ²
        h11(τ) = τ³ - τ²
    """
    
    def __init__(self, num_points: int, output_size: int):
        """
        Args:
            num_points: Number of control points (N)
            output_size: Number of output samples (H or W)
        """
        super().__init__()
        self.num_points = num_points
        self.output_size = output_size
        self.num_segments = num_points - 1
        
        # Pre-compute evaluation grid
        # t ∈ [0, 1] mapped to segments
        t = torch.linspace(0, 1, output_size)
        self.register_buffer('t', t)
        
        # Pre-compute segment indices and local coordinates
        # Scale t to [0, num_segments] to get segment index
        t_scaled = t * self.num_segments
        seg_idx = t_scaled.floor().long().clamp(0, self.num_segments - 1)
        tau = t_scaled - seg_idx.float()  # Local coordinate within segment [0, 1]
        
        self.register_buffer('seg_idx', seg_idx)
        self.register_buffer('tau', tau)
        
        # Pre-compute Hermite basis functions
        tau2 = tau ** 2
        tau3 = tau ** 3
        
        h00 = 2 * tau3 - 3 * tau2 + 1
        h10 = tau3 - 2 * tau2 + tau
        h01 = -2 * tau3 + 3 * tau2
        h11 = tau3 - tau2
        
        self.register_buffer('h00', h00)
        self.register_buffer('h10', h10)
        self.register_buffer('h01', h01)
        self.register_buffer('h11', h11)
    
    def forward(self, control_points: torch.Tensor) -> torch.Tensor:
        """
        Evaluate spline at pre-defined grid points.
        
        Args:
            control_points: [B, R, N] control points for R ranks and N points
            
        Returns:
            spline_values: [B, R, output_size] evaluated spline values
        """
        B, R, N = control_points.shape
        assert N == self.num_points, f"Expected {self.num_points} control points, got {N}"
        
        # Compute tangents using Catmull-Rom formula: m_k = (p_{k+1} - p_{k-1}) / 2
        # For boundary points, use one-sided differences
        tangents = torch.zeros_like(control_points)
        tangents[:, :, 1:-1] = (control_points[:, :, 2:] - control_points[:, :, :-2]) / 2
        tangents[:, :, 0] = control_points[:, :, 1] - control_points[:, :, 0]
        tangents[:, :, -1] = control_points[:, :, -1] - control_points[:, :, -2]
        
        # Scale tangents by segment width (1/num_segments) for proper interpolation
        tangents = tangents / self.num_segments
        
        # Gather control points and tangents for each output position
        # seg_idx: [output_size] -> indices into control_points
        idx_k = self.seg_idx  # [output_size]
        idx_k1 = (idx_k + 1).clamp(max=N-1)  # [output_size]
        
        # Expand indices for batch and rank dimensions
        # [output_size] -> [1, 1, output_size] for gather
        idx_k = idx_k.view(1, 1, -1).expand(B, R, -1)
        idx_k1 = idx_k1.view(1, 1, -1).expand(B, R, -1)
        
        # Gather p_k, p_{k+1}, m_k, m_{k+1}
        p_k = torch.gather(control_points, 2, idx_k)  # [B, R, output_size]
        p_k1 = torch.gather(control_points, 2, idx_k1)
        m_k = torch.gather(tangents, 2, idx_k)
        m_k1 = torch.gather(tangents, 2, idx_k1)
        
        # Evaluate Hermite interpolation
        # h00, h10, h01, h11: [output_size] -> [1, 1, output_size]
        h00 = self.h00.view(1, 1, -1)
        h10 = self.h10.view(1, 1, -1)
        h01 = self.h01.view(1, 1, -1)
        h11 = self.h11.view(1, 1, -1)
        
        result = h00 * p_k + h10 * m_k + h01 * p_k1 + h11 * m_k1
        
        return result


# ============================================================================
# Audio Encoder
# ============================================================================

class AudioEncoder(nn.Module):
    """
    Encode binaural audio (left, right channels) into a latent representation.
    
    Input: [B, 2, H, W] - 2-channel spectrogram (left, right)
    Output: [B, latent_dim] - latent vector
    """
    
    def __init__(
        self,
        input_channels: int = 2,
        latent_dim: int = 512,
        base_channels: int = 64,
        num_layers: int = 5,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Build encoder layers
        layers = []
        in_ch = input_channels
        out_ch = base_channels
        
        for i in range(num_layers):
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_ch = out_ch
            if i < num_layers - 2:
                out_ch = min(out_ch * 2, 512)
        
        self.encoder = nn.Sequential(*layers)
        
        # Adaptive pooling to fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Final projection
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch * 16, latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(latent_dim, latent_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 2, H, W] binaural spectrogram
            
        Returns:
            h: [B, latent_dim] latent representation
        """
        features = self.encoder(x)
        features = self.adaptive_pool(features)
        h = self.fc(features)
        return h


class AudioEncoderUNet(nn.Module):
    """
    UNet-style audio encoder that produces multi-scale features.
    This allows the model to capture both global and local audio cues.
    """
    
    def __init__(
        self,
        input_channels: int = 2,
        latent_dim: int = 512,
        base_channels: int = 64,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder path
        self.enc1 = self._make_enc_block(input_channels, base_channels)
        self.enc2 = self._make_enc_block(base_channels, base_channels * 2)
        self.enc3 = self._make_enc_block(base_channels * 2, base_channels * 4)
        self.enc4 = self._make_enc_block(base_channels * 4, base_channels * 8)
        self.enc5 = self._make_enc_block(base_channels * 8, base_channels * 8)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(base_channels * 8, latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Additional context from intermediate features
        self.context_fc = nn.Sequential(
            nn.Linear(base_channels * 8, latent_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.final_fc = nn.Linear(latent_dim + latent_dim // 2, latent_dim)
    
    def _make_enc_block(self, in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        
        # Global features from bottleneck
        h_global = self.bottleneck(e5)
        
        # Context from e4 (intermediate scale)
        e4_pooled = F.adaptive_avg_pool2d(e4, (1, 1)).flatten(1)
        h_context = self.context_fc(e4_pooled)
        
        # Combine
        h = torch.cat([h_global, h_context], dim=1)
        h = self.final_fc(h)
        
        return h


# ============================================================================
# Spline Depth Head
# ============================================================================

class SplineDepthHead(nn.Module):
    """
    Predict spline control points from latent representation.
    
    Outputs control points for both x and y directions, for each rank.
    The depth is then reconstructed as:
        D(x,y) = Σ_r u_r(x) * v_r(y)
    """
    
    def __init__(
        self,
        latent_dim: int = 512,
        rank: int = 4,
        ctrl_x: int = 8,
        ctrl_y: int = 8,
        output_h: int = 256,
        output_w: int = 256,
    ):
        super().__init__()
        
        self.rank = rank
        self.ctrl_x = ctrl_x
        self.ctrl_y = ctrl_y
        self.output_h = output_h
        self.output_w = output_w
        
        # Shared feature processing
        self.shared_fc = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Separate heads for x and y control points
        self.head_x = nn.Linear(latent_dim, rank * ctrl_x)
        self.head_y = nn.Linear(latent_dim, rank * ctrl_y)
        
        # Optional: learnable bias (base depth level per rank)
        self.bias = nn.Parameter(torch.zeros(rank))
        
        # Spline evaluators
        self.spline_x = CubicHermiteSpline(ctrl_x, output_w)
        self.spline_y = CubicHermiteSpline(ctrl_y, output_h)
        
        # Initialize weights for reasonable depth output
        self._init_weights()
    
    def _init_weights(self):
        # Initialize to produce near-zero output initially
        nn.init.xavier_uniform_(self.head_x.weight, gain=0.1)
        nn.init.zeros_(self.head_x.bias)
        nn.init.xavier_uniform_(self.head_y.weight, gain=0.1)
        nn.init.zeros_(self.head_y.bias)
        
        # Initialize bias to produce reasonable depth (e.g., mid-range)
        # This helps training start from a reasonable depth estimate
        nn.init.constant_(self.bias, 0.5)  # Assuming normalized depth [0, 1]
    
    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            h: [B, latent_dim] latent representation
            
        Returns:
            depth: [B, 1, H, W] predicted depth map
            Px: [B, R, ctrl_x] x-direction control points
            Py: [B, R, ctrl_y] y-direction control points
        """
        B = h.shape[0]
        
        # Process latent
        h = self.shared_fc(h)
        
        # Predict control points
        Px = self.head_x(h).view(B, self.rank, self.ctrl_x)
        Py = self.head_y(h).view(B, self.rank, self.ctrl_y)
        
        # Add base level to first rank (helps convergence)
        # Px[:, 0, :] = Px[:, 0, :] + 1.0  # Removed: let network learn this
        
        # Evaluate splines
        u = self.spline_x(Px)  # [B, R, W]
        v = self.spline_y(Py)  # [B, R, H]
        
        # Low-rank outer product: D(x,y) = Σ_r u_r(x) * v_r(y)
        # u: [B, R, W] -> [B, R, 1, W]
        # v: [B, R, H] -> [B, R, H, 1]
        u = u.unsqueeze(2)  # [B, R, 1, W]
        v = v.unsqueeze(3)  # [B, R, H, 1]
        
        # Outer product for each rank and sum
        # [B, R, 1, W] * [B, R, H, 1] = [B, R, H, W]
        depth_components = u * v
        
        # Add bias per rank: [R] -> [1, R, 1, 1]
        bias = self.bias.view(1, self.rank, 1, 1)
        depth_components = depth_components + bias
        
        # Sum over ranks
        depth = depth_components.sum(dim=1, keepdim=True)  # [B, 1, H, W]
        
        return depth, Px, Py


class SplineDepthHeadMultiScale(nn.Module):
    """
    Multi-scale spline depth head.
    
    Uses coarse-to-fine approach:
    - Coarse level: few control points, captures global structure
    - Fine level: more control points, captures local variations
    
    D(x,y) = D_coarse(x,y) + D_fine(x,y)
    """
    
    def __init__(
        self,
        latent_dim: int = 512,
        rank_coarse: int = 2,
        rank_fine: int = 4,
        ctrl_coarse: int = 4,
        ctrl_fine: int = 8,
        output_h: int = 256,
        output_w: int = 256,
    ):
        super().__init__()
        
        self.rank_coarse = rank_coarse
        self.rank_fine = rank_fine
        
        # Coarse level head
        self.coarse_head = SplineDepthHead(
            latent_dim=latent_dim,
            rank=rank_coarse,
            ctrl_x=ctrl_coarse,
            ctrl_y=ctrl_coarse,
            output_h=output_h,
            output_w=output_w,
        )
        
        # Fine level head
        self.fine_head = SplineDepthHead(
            latent_dim=latent_dim,
            rank=rank_fine,
            ctrl_x=ctrl_fine,
            ctrl_y=ctrl_fine,
            output_h=output_h,
            output_w=output_w,
        )
        
        # Learnable scale for fine details
        self.fine_scale = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            h: [B, latent_dim] latent representation
            
        Returns:
            depth: [B, 1, H, W] predicted depth map
            info: dict with control points and intermediate results
        """
        # Coarse prediction
        depth_coarse, Px_coarse, Py_coarse = self.coarse_head(h)
        
        # Fine prediction
        depth_fine, Px_fine, Py_fine = self.fine_head(h)
        
        # Combine: coarse + scaled fine
        depth = depth_coarse + self.fine_scale * depth_fine
        
        info = {
            'depth_coarse': depth_coarse,
            'depth_fine': depth_fine,
            'Px_coarse': Px_coarse,
            'Py_coarse': Py_coarse,
            'Px_fine': Px_fine,
            'Py_fine': Py_fine,
            'fine_scale': self.fine_scale,
        }
        
        return depth, info


# ============================================================================
# Main Model
# ============================================================================

class AudioSplineDepth(nn.Module):
    """
    Main SplineGS-inspired audio-to-depth model.
    
    Architecture:
        Audio (binaural spectrogram) 
        -> AudioEncoder 
        -> latent h 
        -> SplineDepthHead 
        -> D(x,y) = Σ_r u_r(x) * v_r(y)
    
    The key insight is that audio-based depth estimation is fundamentally
    limited to low-frequency spatial information (due to wavelength constraints).
    Spline representation naturally captures this smooth structure while
    being parameter-efficient and stable to train.
    """
    
    def __init__(
        self,
        cfg=None,
        input_channels: int = 2,
        output_h: int = 256,
        output_w: int = 256,
        latent_dim: int = 512,
        rank: int = 8,
        ctrl_x: int = 8,
        ctrl_y: int = 8,
        encoder_type: str = 'standard',  # 'standard' or 'unet'
        multi_scale: bool = False,
        depth_activation: str = 'none',  # 'none', 'sigmoid', 'softplus'
    ):
        super().__init__()
        
        self.output_h = output_h
        self.output_w = output_w
        self.depth_activation = depth_activation
        self.multi_scale = multi_scale
        
        # Parse config if provided
        if cfg is not None:
            output_h = cfg.dataset.images_size
            output_w = cfg.dataset.images_size
            self.output_h = output_h
            self.output_w = output_w
            self.depth_norm = getattr(cfg.dataset, 'depth_norm', True)
        else:
            self.depth_norm = True
        
        # Audio encoder
        if encoder_type == 'unet':
            self.encoder = AudioEncoderUNet(
                input_channels=input_channels,
                latent_dim=latent_dim,
            )
        else:
            self.encoder = AudioEncoder(
                input_channels=input_channels,
                latent_dim=latent_dim,
            )
        
        # Spline depth head
        if multi_scale:
            self.depth_head = SplineDepthHeadMultiScale(
                latent_dim=latent_dim,
                rank_coarse=max(2, rank // 2),
                rank_fine=rank,
                ctrl_coarse=max(4, ctrl_x // 2),
                ctrl_fine=ctrl_x,
                output_h=output_h,
                output_w=output_w,
            )
        else:
            self.depth_head = SplineDepthHead(
                latent_dim=latent_dim,
                rank=rank,
                ctrl_x=ctrl_x,
                ctrl_y=ctrl_y,
                output_h=output_h,
                output_w=output_w,
            )
        
        # Store config for later use
        self.config = {
            'latent_dim': latent_dim,
            'rank': rank,
            'ctrl_x': ctrl_x,
            'ctrl_y': ctrl_y,
            'encoder_type': encoder_type,
            'multi_scale': multi_scale,
        }
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            audio: [B, 2, H, W] binaural spectrogram
            
        Returns:
            depth: [B, 1, H, W] predicted depth map
        """
        # Encode audio
        h = self.encoder(audio)
        
        # Predict depth via spline decomposition
        if self.multi_scale:
            depth, info = self.depth_head(h)
        else:
            depth, Px, Py = self.depth_head(h)
        
        # Apply activation if specified
        if self.depth_activation == 'sigmoid':
            depth = torch.sigmoid(depth)
        elif self.depth_activation == 'softplus':
            depth = F.softplus(depth)
        # 'none': keep raw output
        
        return depth
    
    def forward_with_info(self, audio: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass with additional information for analysis.
        
        Returns depth map and dictionary with control points, etc.
        """
        h = self.encoder(audio)
        
        if self.multi_scale:
            depth, info = self.depth_head(h)
        else:
            depth, Px, Py = self.depth_head(h)
            info = {'Px': Px, 'Py': Py}
        
        if self.depth_activation == 'sigmoid':
            depth = torch.sigmoid(depth)
        elif self.depth_activation == 'softplus':
            depth = F.softplus(depth)
        
        info['latent'] = h
        
        return depth, info
    
    def get_num_params(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_spline_complexity(self) -> dict:
        """Return spline complexity metrics."""
        if self.multi_scale:
            return {
                'rank_coarse': self.depth_head.rank_coarse,
                'rank_fine': self.depth_head.rank_fine,
                'total_ctrl_points': (
                    self.depth_head.rank_coarse * self.depth_head.coarse_head.ctrl_x * 2 +
                    self.depth_head.rank_fine * self.depth_head.fine_head.ctrl_x * 2
                ),
            }
        else:
            return {
                'rank': self.depth_head.rank,
                'ctrl_x': self.depth_head.ctrl_x,
                'ctrl_y': self.depth_head.ctrl_y,
                'total_ctrl_points': self.depth_head.rank * (
                    self.depth_head.ctrl_x + self.depth_head.ctrl_y
                ),
            }


# ============================================================================
# Model Factory
# ============================================================================

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights."""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    
    print(f'Initialize network with {init_type}')
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize network and move to GPU."""
    if len(gpu_ids) > 0:
        assert torch.cuda.is_available()
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_spline_depth(
    cfg=None,
    input_nc: int = 2,
    output_h: int = 256,
    output_w: int = 256,
    latent_dim: int = 512,
    rank: int = 8,
    ctrl_x: int = 8,
    ctrl_y: int = 8,
    encoder_type: str = 'standard',
    multi_scale: bool = False,
    depth_activation: str = 'none',
    init_type: str = 'normal',
    init_gain: float = 0.02,
    gpu_ids: List[int] = [],
) -> nn.Module:
    """
    Create and initialize AudioSplineDepth model.
    
    Args:
        cfg: Config object with dataset parameters
        input_nc: Number of input channels (2 for binaural)
        output_h: Output height
        output_w: Output width
        latent_dim: Dimension of latent representation
        rank: Number of ranks in low-rank decomposition
        ctrl_x: Number of control points in x direction
        ctrl_y: Number of control points in y direction
        encoder_type: 'standard' or 'unet'
        multi_scale: Use multi-scale spline head
        depth_activation: 'none', 'sigmoid', or 'softplus'
        init_type: Weight initialization type
        init_gain: Initialization gain
        gpu_ids: List of GPU IDs
        
    Returns:
        Initialized model
    """
    net = AudioSplineDepth(
        cfg=cfg,
        input_channels=input_nc,
        output_h=output_h,
        output_w=output_w,
        latent_dim=latent_dim,
        rank=rank,
        ctrl_x=ctrl_x,
        ctrl_y=ctrl_y,
        encoder_type=encoder_type,
        multi_scale=multi_scale,
        depth_activation=depth_activation,
    )
    
    return init_net(net, init_type, init_gain, gpu_ids)


# ============================================================================
# Spline Regularization Losses
# ============================================================================

class SplineSmoothLoss(nn.Module):
    """
    Regularization loss for spline smoothness.
    
    Penalizes large second derivatives (curvature) of control points.
    This encourages smooth depth variations.
    """
    
    def __init__(self, weight: float = 0.01):
        super().__init__()
        self.weight = weight
    
    def forward(self, Px: torch.Tensor, Py: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Px: [B, R, Nx] x-direction control points
            Py: [B, R, Ny] y-direction control points
            
        Returns:
            loss: Smoothness loss
        """
        # Second-order finite difference (curvature approximation)
        if Px.shape[2] >= 3:
            d2_x = Px[:, :, 2:] - 2 * Px[:, :, 1:-1] + Px[:, :, :-2]
            loss_x = (d2_x ** 2).mean()
        else:
            loss_x = 0.0
        
        if Py.shape[2] >= 3:
            d2_y = Py[:, :, 2:] - 2 * Py[:, :, 1:-1] + Py[:, :, :-2]
            loss_y = (d2_y ** 2).mean()
        else:
            loss_y = 0.0
        
        return self.weight * (loss_x + loss_y)


class SplineSparsityLoss(nn.Module):
    """
    Sparsity regularization for control point magnitudes.
    
    Encourages some control points to be near zero, promoting
    adaptive complexity similar to MACP pruning.
    """
    
    def __init__(self, weight: float = 0.001):
        super().__init__()
        self.weight = weight
    
    def forward(self, Px: torch.Tensor, Py: torch.Tensor) -> torch.Tensor:
        """L1 sparsity on control point magnitudes."""
        loss = Px.abs().mean() + Py.abs().mean()
        return self.weight * loss


# ============================================================================
# Testing
# ============================================================================

if __name__ == '__main__':
    # Test the model
    print("Testing AudioSplineDepth model...")
    
    # Create model
    model = AudioSplineDepth(
        input_channels=2,
        output_h=256,
        output_w=256,
        latent_dim=512,
        rank=8,
        ctrl_x=8,
        ctrl_y=8,
    )
    
    # Test forward pass
    batch_size = 4
    audio = torch.randn(batch_size, 2, 256, 256)
    
    depth = model(audio)
    print(f"Input shape: {audio.shape}")
    print(f"Output shape: {depth.shape}")
    print(f"Depth range: [{depth.min().item():.4f}, {depth.max().item():.4f}]")
    print(f"Number of parameters: {model.get_num_params():,}")
    print(f"Spline complexity: {model.get_spline_complexity()}")
    
    # Test with info
    depth, info = model.forward_with_info(audio)
    print(f"\nControl points Px shape: {info['Px'].shape}")
    print(f"Control points Py shape: {info['Py'].shape}")
    print(f"Latent shape: {info['latent'].shape}")
    
    # Test multi-scale version
    print("\n\nTesting Multi-Scale AudioSplineDepth model...")
    model_ms = AudioSplineDepth(
        input_channels=2,
        output_h=256,
        output_w=256,
        latent_dim=512,
        rank=8,
        ctrl_x=8,
        ctrl_y=8,
        multi_scale=True,
    )
    
    depth_ms, info_ms = model_ms.forward_with_info(audio)
    print(f"Multi-scale output shape: {depth_ms.shape}")
    print(f"Coarse depth shape: {info_ms['depth_coarse'].shape}")
    print(f"Fine depth shape: {info_ms['depth_fine'].shape}")
    print(f"Fine scale: {info_ms['fine_scale'].item():.4f}")
    print(f"Number of parameters: {model_ms.get_num_params():,}")
    
    print("\nAll tests passed!")


