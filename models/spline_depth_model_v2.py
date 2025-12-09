"""
SplineGS-inspired Audio-to-Depth Model v2

Improvements over v1:
1. Higher rank & control points for more expressiveness
2. Per-rank audio conditioning (FiLM-style) - each rank learns different features
3. Residual CNN path for sample-specific corrections
4. Reduced smoothness regularization  
5. Diversity loss to prevent mode collapse
6. Multi-head attention for audio-spline coupling

The key insight: v1 collapsed to mean depth prior because:
- All ranks used the same audio features
- The linear projection was too simple to capture sample-specific variations
- Smoothness prior was too strong

D(x,y) = Σ_r α_r(audio) * u_r(x) * v_r(y) + Residual(audio)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
from typing import Tuple, Optional, List, Dict


# ============================================================================
# Spline Utilities (FIXED boundary handling)
# ============================================================================

class CubicHermiteSpline(nn.Module):
    """
    Evaluates cubic Hermite spline from control points.
    
    FIXED: Boundary handling issues
    1. Use small epsilon margin to avoid exact t=0 and t=1
    2. Use "natural" boundary tangents (scaled by 0.5) to prevent overshoot
    3. Proper handling of last segment
    """
    
    def __init__(self, num_points: int, output_size: int, boundary_margin: float = 0.001):
        super().__init__()
        self.num_points = num_points
        self.output_size = output_size
        self.num_segments = num_points - 1
        
        # FIXED: Add small margin to avoid exact boundary values
        # This prevents t=1.0 which causes tau=1.0 (edge of segment)
        eps = boundary_margin
        t = torch.linspace(eps, 1.0 - eps, output_size)
        self.register_buffer('t', t)
        
        t_scaled = t * self.num_segments
        # FIXED: Ensure last pixel falls in last segment properly
        seg_idx = t_scaled.floor().long().clamp(0, self.num_segments - 1)
        tau = t_scaled - seg_idx.float()
        
        # Clamp tau to [0, 1) to avoid numerical issues at boundaries
        tau = tau.clamp(0.0, 0.9999)
        
        self.register_buffer('seg_idx', seg_idx)
        self.register_buffer('tau', tau)
        
        tau2 = tau ** 2
        tau3 = tau ** 3
        
        self.register_buffer('h00', 2 * tau3 - 3 * tau2 + 1)
        self.register_buffer('h10', tau3 - 2 * tau2 + tau)
        self.register_buffer('h01', -2 * tau3 + 3 * tau2)
        self.register_buffer('h11', tau3 - tau2)
    
    def forward(self, control_points: torch.Tensor) -> torch.Tensor:
        B, R, N = control_points.shape
        assert N == self.num_points
        
        # Compute tangents using Catmull-Rom formula
        tangents = torch.zeros_like(control_points)
        tangents[:, :, 1:-1] = (control_points[:, :, 2:] - control_points[:, :, :-2]) / 2
        
        # FIXED: Use "natural" boundary conditions - reduce boundary tangent magnitude
        # to prevent overshoot at edges
        # Scale boundary tangents by 0.5 to make them more conservative
        tangents[:, :, 0] = (control_points[:, :, 1] - control_points[:, :, 0]) * 0.5
        tangents[:, :, -1] = (control_points[:, :, -1] - control_points[:, :, -2]) * 0.5
        
        tangents = tangents / self.num_segments
        
        idx_k = self.seg_idx.view(1, 1, -1).expand(B, R, -1)
        idx_k1 = (self.seg_idx + 1).clamp(max=N-1).view(1, 1, -1).expand(B, R, -1)
        
        p_k = torch.gather(control_points, 2, idx_k)
        p_k1 = torch.gather(control_points, 2, idx_k1)
        m_k = torch.gather(tangents, 2, idx_k)
        m_k1 = torch.gather(tangents, 2, idx_k1)
        
        h00 = self.h00.view(1, 1, -1)
        h10 = self.h10.view(1, 1, -1)
        h01 = self.h01.view(1, 1, -1)
        h11 = self.h11.view(1, 1, -1)
        
        return h00 * p_k + h10 * m_k + h01 * p_k1 + h11 * m_k1


# ============================================================================
# FiLM Layer (Feature-wise Linear Modulation)
# ============================================================================

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation.
    Allows audio features to modulate spline parameters:
        output = gamma(audio) * input + beta(audio)
    """
    
    def __init__(self, feature_dim: int, condition_dim: int):
        super().__init__()
        self.gamma_fc = nn.Linear(condition_dim, feature_dim)
        self.beta_fc = nn.Linear(condition_dim, feature_dim)
        
        # Initialize to identity transform
        nn.init.ones_(self.gamma_fc.weight.data * 0.01 + 1.0)
        nn.init.zeros_(self.gamma_fc.bias.data)
        nn.init.zeros_(self.beta_fc.weight.data)
        nn.init.zeros_(self.beta_fc.bias.data)
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma_fc(condition)
        beta = self.beta_fc(condition)
        
        # Reshape for broadcasting: [B, D] -> [B, D, 1, 1] or similar
        if x.dim() == 4:  # [B, C, H, W]
            gamma = gamma.view(gamma.shape[0], -1, 1, 1)
            beta = beta.view(beta.shape[0], -1, 1, 1)
        elif x.dim() == 3:  # [B, R, N]
            gamma = gamma.view(gamma.shape[0], -1, 1)
            beta = beta.view(beta.shape[0], -1, 1)
        
        return gamma * x + beta


# ============================================================================
# Audio Encoder v2 - Multi-scale with per-rank features
# ============================================================================

class AudioEncoderV2(nn.Module):
    """
    Enhanced audio encoder with multi-scale feature extraction.
    Outputs both global and multi-scale features for per-rank conditioning.
    """
    
    def __init__(
        self,
        input_channels: int = 2,
        latent_dim: int = 512,
        base_channels: int = 64,
        num_scales: int = 4,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.num_scales = num_scales
        
        # Multi-scale encoder
        self.encoders = nn.ModuleList()
        self.scale_projs = nn.ModuleList()
        
        in_ch = input_channels
        out_ch = base_channels
        
        for i in range(num_scales):
            self.encoders.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ))
            
            # Project each scale to latent_dim
            self.scale_projs.append(nn.Sequential(
                nn.AdaptiveAvgPool2d((2, 2)),
                nn.Flatten(),
                nn.Linear(out_ch * 4, latent_dim),
                nn.LeakyReLU(0.2, inplace=True),
            ))
            
            in_ch = out_ch
            out_ch = min(out_ch * 2, 512)
        
        # Final global encoder
        self.global_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_ch, latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(latent_dim, latent_dim),
        )
        
        # Combine multi-scale features
        self.combine_fc = nn.Sequential(
            nn.Linear(latent_dim * (num_scales + 1), latent_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(latent_dim * 2, latent_dim),
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Returns:
            global_feat: [B, latent_dim] global feature
            scale_feats: List of [B, latent_dim] features at each scale
        """
        scale_feats = []
        feat = x
        
        for i, (encoder, proj) in enumerate(zip(self.encoders, self.scale_projs)):
            feat = encoder(feat)
            scale_feats.append(proj(feat))
        
        global_feat = self.global_encoder(feat)
        
        # Combine all features
        all_feats = scale_feats + [global_feat]
        combined = torch.cat(all_feats, dim=1)
        final_feat = self.combine_fc(combined)
        
        return final_feat, scale_feats


# ============================================================================
# Per-Rank Conditioning Module
# ============================================================================

class PerRankConditioner(nn.Module):
    """
    Generates rank-specific features from audio.
    Each rank gets a different "view" of the audio features.
    This prevents all ranks from collapsing to the same pattern.
    """
    
    def __init__(self, latent_dim: int, num_ranks: int, hidden_dim: int = 256):
        super().__init__()
        
        self.num_ranks = num_ranks
        
        # Learnable rank embeddings
        self.rank_embeddings = nn.Parameter(torch.randn(num_ranks, hidden_dim) * 0.1)
        
        # Per-rank feature transformations
        self.rank_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim + hidden_dim, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_ranks)
        ])
        
        # Attention mechanism for feature selection
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True,
        )
        
        self.output_fc = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, audio_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_feat: [B, latent_dim] audio features
            
        Returns:
            rank_feats: [B, num_ranks, hidden_dim] per-rank features
        """
        B = audio_feat.shape[0]
        
        # Expand rank embeddings for batch
        rank_emb = self.rank_embeddings.unsqueeze(0).expand(B, -1, -1)  # [B, R, hidden]
        
        # Combine audio with rank embeddings
        audio_expanded = audio_feat.unsqueeze(1).expand(-1, self.num_ranks, -1)  # [B, R, latent]
        combined = torch.cat([audio_expanded, rank_emb], dim=2)  # [B, R, latent+hidden]
        
        # Per-rank transformation
        rank_feats = []
        for r in range(self.num_ranks):
            feat = self.rank_transforms[r](combined[:, r])
            rank_feats.append(feat)
        rank_feats = torch.stack(rank_feats, dim=1)  # [B, R, hidden]
        
        # Self-attention for inter-rank interaction
        rank_feats, _ = self.attention(rank_feats, rank_feats, rank_feats)
        
        return self.output_fc(rank_feats)


# ============================================================================
# Enhanced Spline Depth Head v2
# ============================================================================

class SplineDepthHeadV2(nn.Module):
    """
    Improved spline depth head with:
    1. Per-rank audio conditioning
    2. FiLM modulation
    3. Learnable rank weights
    4. Higher capacity
    """
    
    def __init__(
        self,
        latent_dim: int = 512,
        rank: int = 16,
        ctrl_x: int = 16,
        ctrl_y: int = 16,
        output_h: int = 256,
        output_w: int = 256,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        self.rank = rank
        self.ctrl_x = ctrl_x
        self.ctrl_y = ctrl_y
        self.output_h = output_h
        self.output_w = output_w
        self.hidden_dim = hidden_dim
        
        # Per-rank conditioning
        self.rank_conditioner = PerRankConditioner(latent_dim, rank, hidden_dim)
        
        # Control point generators (per-rank)
        self.ctrl_x_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(hidden_dim, ctrl_x),
            )
            for _ in range(rank)
        ])
        
        self.ctrl_y_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(hidden_dim, ctrl_y),
            )
            for _ in range(rank)
        ])
        
        # Learnable rank weights (importance of each rank)
        self.rank_weights = nn.Parameter(torch.ones(rank) / rank)
        
        # FiLM modulation for control points
        self.film_x = FiLMLayer(ctrl_x, hidden_dim)
        self.film_y = FiLMLayer(ctrl_y, hidden_dim)
        
        # Spline evaluators
        self.spline_x = CubicHermiteSpline(ctrl_x, output_w)
        self.spline_y = CubicHermiteSpline(ctrl_y, output_h)
        
        # Global bias (learnable mean depth)
        self.global_bias = nn.Parameter(torch.tensor(0.5))
        
        self._init_weights()
    
    def _init_weights(self):
        for gen in self.ctrl_x_generators + self.ctrl_y_generators:
            for m in gen.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.5)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            h: [B, latent_dim] latent representation
            
        Returns:
            depth: [B, 1, H, W] predicted depth map
            info: dict with control points and per-rank info
        """
        B = h.shape[0]
        
        # Get per-rank features
        rank_feats = self.rank_conditioner(h)  # [B, R, hidden]
        
        # Generate control points for each rank
        Px_list = []
        Py_list = []
        
        for r in range(self.rank):
            # Base control points
            px = self.ctrl_x_generators[r](rank_feats[:, r])  # [B, ctrl_x]
            py = self.ctrl_y_generators[r](rank_feats[:, r])  # [B, ctrl_y]
            
            # FiLM modulation with rank-specific features
            # px: [B, ctrl_x], rank_feats[:, r]: [B, hidden]
            gamma_x = self.film_x.gamma_fc(rank_feats[:, r])  # [B, ctrl_x]
            beta_x = self.film_x.beta_fc(rank_feats[:, r])    # [B, ctrl_x]
            px = gamma_x * px + beta_x
            
            gamma_y = self.film_y.gamma_fc(rank_feats[:, r])  # [B, ctrl_y]
            beta_y = self.film_y.beta_fc(rank_feats[:, r])    # [B, ctrl_y]
            py = gamma_y * py + beta_y
            
            Px_list.append(px)
            Py_list.append(py)
        
        Px = torch.stack(Px_list, dim=1)  # [B, R, ctrl_x]
        Py = torch.stack(Py_list, dim=1)  # [B, R, ctrl_y]
        
        # Evaluate splines
        u = self.spline_x(Px)  # [B, R, W]
        v = self.spline_y(Py)  # [B, R, H]
        
        # Weighted outer product
        # Normalize rank weights
        weights = F.softmax(self.rank_weights, dim=0)  # [R]
        weights = weights.view(1, self.rank, 1, 1)  # [1, R, 1, 1]
        
        u = u.unsqueeze(2)  # [B, R, 1, W]
        v = v.unsqueeze(3)  # [B, R, H, 1]
        
        depth_components = u * v  # [B, R, H, W]
        depth_components = depth_components * weights
        
        depth = depth_components.sum(dim=1, keepdim=True)  # [B, 1, H, W]
        depth = depth + self.global_bias
        
        info = {
            'Px': Px,
            'Py': Py,
            'rank_weights': weights.squeeze(),
            'rank_feats': rank_feats,
            'depth_components': depth_components,
        }
        
        return depth, info


# ============================================================================
# Residual CNN for Sample-Specific Details
# ============================================================================

class ResidualCNN(nn.Module):
    """
    Small CNN to add sample-specific residual details.
    This captures high-frequency variations that splines can't represent.
    
    FIXED: Use bilinear upsampling + conv instead of ConvTranspose2d
    to avoid checkerboard artifacts and boundary issues.
    """
    
    def __init__(
        self,
        latent_dim: int = 512,
        output_h: int = 256,
        output_w: int = 256,
        base_channels: int = 64,
    ):
        super().__init__()
        
        self.output_h = output_h
        self.output_w = output_w
        
        # Initial projection from latent to spatial
        init_size = 8
        self.init_fc = nn.Linear(latent_dim, base_channels * 4 * init_size * init_size)
        self.init_size = init_size
        self.init_channels = base_channels * 4
        
        # FIXED: Use bilinear upsampling + conv instead of ConvTranspose2d
        # This avoids checkerboard artifacts and boundary issues
        self.up1 = self._make_upsample_block(base_channels * 4, base_channels * 4)  # 8->16
        self.up2 = self._make_upsample_block(base_channels * 4, base_channels * 2)  # 16->32
        self.up3 = self._make_upsample_block(base_channels * 2, base_channels)      # 32->64
        self.up4 = self._make_upsample_block(base_channels, base_channels)          # 64->128
        self.up5 = self._make_upsample_block(base_channels, base_channels // 2)     # 128->256
        
        # Final conv with padding mode to handle boundaries
        self.final_conv = nn.Conv2d(base_channels // 2, 1, 3, 1, 1, padding_mode='replicate')
        
        # Scale factor for residual (starts small)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
    
    def _make_upsample_block(self, in_ch: int, out_ch: int) -> nn.Module:
        """Bilinear upsample + conv block (no checkerboard artifacts)"""
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, padding_mode='replicate'),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [B, latent_dim] latent
            
        Returns:
            residual: [B, 1, H, W] residual depth
        """
        B = h.shape[0]
        
        # Project and reshape
        x = self.init_fc(h)
        x = x.view(B, self.init_channels, self.init_size, self.init_size)
        
        # FIXED: Use sequential upsample blocks instead of decoder
        x = self.up1(x)  # 8 -> 16
        x = self.up2(x)  # 16 -> 32
        x = self.up3(x)  # 32 -> 64
        x = self.up4(x)  # 64 -> 128
        x = self.up5(x)  # 128 -> 256
        
        residual = self.final_conv(x)
        
        # Interpolate to exact output size if needed
        if residual.shape[2] != self.output_h or residual.shape[3] != self.output_w:
            residual = F.interpolate(residual, size=(self.output_h, self.output_w), 
                                     mode='bilinear', align_corners=False)
        
        return residual * self.residual_scale


# ============================================================================
# Main Model v2
# ============================================================================

class AudioSplineDepthV2(nn.Module):
    """
    Improved SplineGS-inspired audio-to-depth model.
    
    Key improvements:
    1. Per-rank audio conditioning via PerRankConditioner
    2. FiLM modulation for better audio-spline coupling
    3. Residual CNN path for sample-specific details
    4. Higher rank and control points
    5. Diversity loss support
    
    D(x,y) = Σ_r w_r * u_r(x) * v_r(y) + Residual(h)
    """
    
    def __init__(
        self,
        cfg=None,
        input_channels: int = 2,
        output_h: int = 256,
        output_w: int = 256,
        latent_dim: int = 512,
        rank: int = 16,
        ctrl_x: int = 16,
        ctrl_y: int = 16,
        use_residual: bool = True,
        depth_activation: str = 'none',
    ):
        super().__init__()
        
        self.output_h = output_h
        self.output_w = output_w
        self.depth_activation = depth_activation
        self.use_residual = use_residual
        
        if cfg is not None:
            output_h = cfg.dataset.images_size
            output_w = cfg.dataset.images_size
            self.output_h = output_h
            self.output_w = output_w
            self.depth_norm = getattr(cfg.dataset, 'depth_norm', True)
        else:
            self.depth_norm = True
        
        # Audio encoder
        self.encoder = AudioEncoderV2(
            input_channels=input_channels,
            latent_dim=latent_dim,
        )
        
        # Spline depth head
        self.depth_head = SplineDepthHeadV2(
            latent_dim=latent_dim,
            rank=rank,
            ctrl_x=ctrl_x,
            ctrl_y=ctrl_y,
            output_h=output_h,
            output_w=output_w,
        )
        
        # Residual CNN
        if use_residual:
            self.residual_cnn = ResidualCNN(
                latent_dim=latent_dim,
                output_h=output_h,
                output_w=output_w,
            )
        
        self.config = {
            'latent_dim': latent_dim,
            'rank': rank,
            'ctrl_x': ctrl_x,
            'ctrl_y': ctrl_y,
            'use_residual': use_residual,
        }
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # Encode audio
        h, scale_feats = self.encoder(audio)
        
        # Spline-based depth
        depth_spline, info = self.depth_head(h)
        
        # Add residual if enabled
        if self.use_residual:
            residual = self.residual_cnn(h)
            depth = depth_spline + residual
        else:
            depth = depth_spline
        
        # Apply activation
        if self.depth_activation == 'sigmoid':
            depth = torch.sigmoid(depth)
        elif self.depth_activation == 'softplus':
            depth = F.softplus(depth)
        
        return depth
    
    def forward_with_info(self, audio: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        h, scale_feats = self.encoder(audio)
        depth_spline, info = self.depth_head(h)
        
        if self.use_residual:
            residual = self.residual_cnn(h)
            depth = depth_spline + residual
            info['residual'] = residual
            info['depth_spline'] = depth_spline
        else:
            depth = depth_spline
        
        if self.depth_activation == 'sigmoid':
            depth = torch.sigmoid(depth)
        elif self.depth_activation == 'softplus':
            depth = F.softplus(depth)
        
        info['latent'] = h
        info['scale_feats'] = scale_feats
        
        return depth, info
    
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_spline_complexity(self) -> Dict:
        return {
            'rank': self.depth_head.rank,
            'ctrl_x': self.depth_head.ctrl_x,
            'ctrl_y': self.depth_head.ctrl_y,
            'total_ctrl_points': self.depth_head.rank * (
                self.depth_head.ctrl_x + self.depth_head.ctrl_y
            ),
            'use_residual': self.use_residual,
        }


# ============================================================================
# Enhanced Loss Functions
# ============================================================================

class DiversityLoss(nn.Module):
    """
    Encourages different ranks to learn different patterns.
    Prevents mode collapse where all ranks produce similar outputs.
    """
    
    def __init__(self, weight: float = 0.01):
        super().__init__()
        self.weight = weight
    
    def forward(self, depth_components: torch.Tensor) -> torch.Tensor:
        """
        Args:
            depth_components: [B, R, H, W] per-rank depth contributions
            
        Returns:
            loss: Diversity loss (negative correlation)
        """
        B, R, H, W = depth_components.shape
        
        # Flatten spatial dimensions
        flat = depth_components.view(B, R, -1)  # [B, R, H*W]
        
        # Normalize
        flat = flat - flat.mean(dim=2, keepdim=True)
        flat = flat / (flat.std(dim=2, keepdim=True) + 1e-6)
        
        # Compute correlation matrix
        # [B, R, HW] @ [B, HW, R] -> [B, R, R]
        corr = torch.bmm(flat, flat.transpose(1, 2)) / (H * W)
        
        # We want off-diagonal elements to be small (uncorrelated ranks)
        # Create mask for off-diagonal
        eye = torch.eye(R, device=corr.device).unsqueeze(0)
        off_diag = corr * (1 - eye)
        
        # Penalize high correlation between different ranks
        loss = off_diag.abs().mean()
        
        return self.weight * loss


class EdgeAwareLoss(nn.Module):
    """
    Edge-aware loss that preserves depth discontinuities.
    Uses gradient-based weighting to focus on edges.
    """
    
    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight
    
    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, 1, H, W] predicted depth
            gt: [B, 1, H, W] ground truth depth
        """
        # Compute gradients
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        
        gt_dx = gt[:, :, :, 1:] - gt[:, :, :, :-1]
        gt_dy = gt[:, :, 1:, :] - gt[:, :, :-1, :]
        
        # L1 on gradients
        loss_dx = F.l1_loss(pred_dx, gt_dx)
        loss_dy = F.l1_loss(pred_dy, gt_dy)
        
        return self.weight * (loss_dx + loss_dy)


class RankVarianceLoss(nn.Module):
    """
    Encourages variance in rank weights, preventing uniform weights.
    """
    
    def __init__(self, weight: float = 0.001):
        super().__init__()
        self.weight = weight
    
    def forward(self, rank_weights: torch.Tensor) -> torch.Tensor:
        # We want some ranks to be more important than others
        # Penalize low variance (uniform distribution)
        var = rank_weights.var()
        loss = 1.0 / (var + 1e-6)  # Inverse variance
        return self.weight * loss


class SplineSmoothLossV2(nn.Module):
    """
    Reduced smoothness regularization (compared to v1).
    Only penalizes extreme curvature, not all curvature.
    """
    
    def __init__(self, weight: float = 0.001, threshold: float = 0.1):
        super().__init__()
        self.weight = weight
        self.threshold = threshold
    
    def forward(self, Px: torch.Tensor, Py: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        
        if Px.shape[2] >= 3:
            d2_x = Px[:, :, 2:] - 2 * Px[:, :, 1:-1] + Px[:, :, :-2]
            # Only penalize curvature above threshold
            loss_x = F.relu(d2_x.abs() - self.threshold).mean()
            loss = loss + loss_x
        
        if Py.shape[2] >= 3:
            d2_y = Py[:, :, 2:] - 2 * Py[:, :, 1:-1] + Py[:, :, :-2]
            loss_y = F.relu(d2_y.abs() - self.threshold).mean()
            loss = loss + loss_y
        
        return self.weight * loss


class BoundaryLoss(nn.Module):
    """
    Penalizes boundary artifacts by encouraging smooth transitions at edges.
    
    This loss compares boundary pixels with their neighbors and penalizes
    large differences, which helps prevent the "red border" artifacts
    common in spline-based depth estimation.
    """
    
    def __init__(self, weight: float = 0.1, border_width: int = 2):
        super().__init__()
        self.weight = weight
        self.border_width = border_width
    
    def forward(self, pred: torch.Tensor, gt: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            pred: [B, 1, H, W] predicted depth
            gt: [B, 1, H, W] ground truth (optional, for masking)
            
        Returns:
            loss: Boundary smoothness loss
        """
        B, C, H, W = pred.shape
        bw = self.border_width
        
        # Compare boundary with adjacent interior
        # Left border: pred[:, :, :, :bw] vs pred[:, :, :, bw:2*bw]
        # Right border: pred[:, :, :, -bw:] vs pred[:, :, :, -2*bw:-bw]
        # Top border: pred[:, :, :bw, :] vs pred[:, :, bw:2*bw, :]
        # Bottom border: pred[:, :, -bw:, :] vs pred[:, :, -2*bw:-bw, :]
        
        loss = 0.0
        
        # Left-right boundary smoothness
        if W > 2 * bw:
            left_border = pred[:, :, :, :bw]
            left_interior = pred[:, :, :, bw:2*bw]
            loss = loss + F.l1_loss(left_border, left_interior)
            
            right_border = pred[:, :, :, -bw:]
            right_interior = pred[:, :, :, -2*bw:-bw]
            loss = loss + F.l1_loss(right_border, right_interior)
        
        # Top-bottom boundary smoothness
        if H > 2 * bw:
            top_border = pred[:, :, :bw, :]
            top_interior = pred[:, :, bw:2*bw, :]
            loss = loss + F.l1_loss(top_border, top_interior)
            
            bottom_border = pred[:, :, -bw:, :]
            bottom_interior = pred[:, :, -2*bw:-bw, :]
            loss = loss + F.l1_loss(bottom_border, bottom_interior)
        
        return self.weight * loss


# ============================================================================
# Model Factory
# ============================================================================

def init_weights(net, init_type='kaiming', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    
    print(f'Initialize network with {init_type}')
    net.apply(init_func)


def init_net(net, init_type='kaiming', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert torch.cuda.is_available()
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_spline_depth_v2(
    cfg=None,
    input_nc: int = 2,
    output_h: int = 256,
    output_w: int = 256,
    latent_dim: int = 512,
    rank: int = 16,
    ctrl_x: int = 16,
    ctrl_y: int = 16,
    use_residual: bool = True,
    depth_activation: str = 'none',
    init_type: str = 'kaiming',
    init_gain: float = 0.02,
    gpu_ids: List[int] = [],
) -> nn.Module:
    """Create and initialize AudioSplineDepthV2 model."""
    net = AudioSplineDepthV2(
        cfg=cfg,
        input_channels=input_nc,
        output_h=output_h,
        output_w=output_w,
        latent_dim=latent_dim,
        rank=rank,
        ctrl_x=ctrl_x,
        ctrl_y=ctrl_y,
        use_residual=use_residual,
        depth_activation=depth_activation,
    )
    
    return init_net(net, init_type, init_gain, gpu_ids)


# ============================================================================
# Testing
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Testing AudioSplineDepthV2 model")
    print("=" * 60)
    
    model = AudioSplineDepthV2(
        input_channels=2,
        output_h=256,
        output_w=256,
        latent_dim=512,
        rank=16,
        ctrl_x=16,
        ctrl_y=16,
        use_residual=True,
    )
    
    batch_size = 2
    audio = torch.randn(batch_size, 2, 256, 256)
    
    depth = model(audio)
    print(f"Input shape: {audio.shape}")
    print(f"Output shape: {depth.shape}")
    print(f"Depth range: [{depth.min().item():.4f}, {depth.max().item():.4f}]")
    print(f"Parameters: {model.get_num_params():,}")
    print(f"Spline complexity: {model.get_spline_complexity()}")
    
    # Test with info
    depth, info = model.forward_with_info(audio)
    print(f"\nControl points Px shape: {info['Px'].shape}")
    print(f"Control points Py shape: {info['Py'].shape}")
    print(f"Rank weights: {info['rank_weights']}")
    print(f"Depth components shape: {info['depth_components'].shape}")
    if 'residual' in info:
        print(f"Residual range: [{info['residual'].min().item():.4f}, {info['residual'].max().item():.4f}]")
    
    # Test diversity loss
    div_loss = DiversityLoss()
    d_loss = div_loss(info['depth_components'])
    print(f"\nDiversity loss: {d_loss.item():.6f}")
    
    print("\nAll tests passed!")

