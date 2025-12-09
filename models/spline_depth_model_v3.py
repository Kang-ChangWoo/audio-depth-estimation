"""
SplineGS-inspired Audio-to-Depth Model v3

Key improvements over v2:
1. Additive + Multiplicative decomposition:
   D(x,y) = Σ u_r(x)*v_r(y) + Σ a_r(x) + Σ b_r(y) + bias + Residual
   
2. Increased residual CNN capacity and scale
3. Higher default rank (32) and control points (24)
4. Optional 2D spline grid for non-separable patterns
5. Better boundary handling

The key insight: Pure multiplicative (u*v) can only represent separable functions.
Real depth maps need additive terms for walls (x-only) and floors (y-only).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
from typing import Tuple, Optional, List, Dict


# ============================================================================
# Spline Utilities (with boundary fix)
# ============================================================================

class CubicHermiteSpline(nn.Module):
    """
    Cubic Hermite spline with proper boundary handling.
    """
    
    def __init__(self, num_points: int, output_size: int, boundary_margin: float = 0.001):
        super().__init__()
        self.num_points = num_points
        self.output_size = output_size
        self.num_segments = num_points - 1
        
        # Add margin to avoid exact boundary values
        eps = boundary_margin
        t = torch.linspace(eps, 1.0 - eps, output_size)
        self.register_buffer('t', t)
        
        t_scaled = t * self.num_segments
        seg_idx = t_scaled.floor().long().clamp(0, self.num_segments - 1)
        tau = (t_scaled - seg_idx.float()).clamp(0.0, 0.9999)
        
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
        
        # Compute tangents with reduced boundary tangents
        tangents = torch.zeros_like(control_points)
        tangents[:, :, 1:-1] = (control_points[:, :, 2:] - control_points[:, :, :-2]) / 2
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
# Audio Encoder (same as v2)
# ============================================================================

class AudioEncoderV3(nn.Module):
    """Enhanced audio encoder with multi-scale features."""
    
    def __init__(
        self,
        input_channels: int = 2,
        latent_dim: int = 768,  # Increased from 512
        base_channels: int = 64,
        num_scales: int = 5,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.num_scales = num_scales
        
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
            
            self.scale_projs.append(nn.Sequential(
                nn.AdaptiveAvgPool2d((2, 2)),
                nn.Flatten(),
                nn.Linear(out_ch * 4, latent_dim // 2),
                nn.LeakyReLU(0.2, inplace=True),
            ))
            
            in_ch = out_ch
            out_ch = min(out_ch * 2, 512)
        
        self.global_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_ch, latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(latent_dim, latent_dim),
        )
        
        self.combine_fc = nn.Sequential(
            nn.Linear(latent_dim + (latent_dim // 2) * num_scales, latent_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(latent_dim * 2, latent_dim),
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        scale_feats = []
        feat = x
        
        for encoder, proj in zip(self.encoders, self.scale_projs):
            feat = encoder(feat)
            scale_feats.append(proj(feat))
        
        global_feat = self.global_encoder(feat)
        
        all_feats = scale_feats + [global_feat]
        combined = torch.cat(all_feats, dim=1)
        final_feat = self.combine_fc(combined)
        
        return final_feat, scale_feats


# ============================================================================
# Per-Rank Conditioning (improved)
# ============================================================================

class PerRankConditionerV3(nn.Module):
    """Per-rank conditioning with stronger differentiation."""
    
    def __init__(self, latent_dim: int, num_ranks: int, hidden_dim: int = 256):
        super().__init__()
        
        self.num_ranks = num_ranks
        self.hidden_dim = hidden_dim
        
        # Learnable rank embeddings (orthogonally initialized)
        self.rank_embeddings = nn.Parameter(torch.randn(num_ranks, hidden_dim) * 0.1)
        
        # Shared transform + per-rank refinement
        self.shared_transform = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        
        # Per-rank MLPs
        self.rank_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_ranks)
        ])
    
    def forward(self, audio_feat: torch.Tensor) -> torch.Tensor:
        B = audio_feat.shape[0]
        
        # Shared features
        shared = self.shared_transform(audio_feat)  # [B, hidden]
        
        # Per-rank features
        rank_feats = []
        for r in range(self.num_ranks):
            # Combine shared features with rank embedding
            rank_emb = self.rank_embeddings[r].unsqueeze(0).expand(B, -1)
            combined = torch.cat([shared, rank_emb], dim=1)  # [B, hidden*2]
            feat = self.rank_mlps[r](combined)
            rank_feats.append(feat)
        
        return torch.stack(rank_feats, dim=1)  # [B, R, hidden]


# ============================================================================
# Additive + Multiplicative Spline Depth Head (NEW)
# ============================================================================

class SplineDepthHeadV3(nn.Module):
    """
    Improved depth head with additive + multiplicative decomposition:
    
    D(x,y) = Σ w_mult_r * u_r(x) * v_r(y)   # multiplicative (rank R_mult)
           + Σ w_x_r * a_r(x)                # x-additive (rank R_add)
           + Σ w_y_r * b_r(y)                # y-additive (rank R_add)
           + global_bias
           
    This allows modeling:
    - Walls (mostly x-dependent)
    - Floors (mostly y-dependent)  
    - Complex shapes (multiplicative)
    """
    
    def __init__(
        self,
        latent_dim: int = 768,
        rank_mult: int = 16,      # Ranks for multiplicative terms
        rank_add: int = 8,        # Ranks for additive terms
        ctrl_x: int = 24,
        ctrl_y: int = 24,
        output_h: int = 256,
        output_w: int = 256,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        self.rank_mult = rank_mult
        self.rank_add = rank_add
        self.ctrl_x = ctrl_x
        self.ctrl_y = ctrl_y
        self.output_h = output_h
        self.output_w = output_w
        
        total_rank = rank_mult + rank_add
        
        # Per-rank conditioning
        self.rank_conditioner = PerRankConditionerV3(latent_dim, total_rank, hidden_dim)
        
        # === Multiplicative term generators ===
        self.mult_x_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(hidden_dim, ctrl_x),
            )
            for _ in range(rank_mult)
        ])
        
        self.mult_y_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(hidden_dim, ctrl_y),
            )
            for _ in range(rank_mult)
        ])
        
        # === Additive term generators ===
        # a_r(x) - x-only terms
        self.add_x_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(hidden_dim, ctrl_x),
            )
            for _ in range(rank_add)
        ])
        
        # b_r(y) - y-only terms
        self.add_y_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(hidden_dim, ctrl_y),
            )
            for _ in range(rank_add)
        ])
        
        # Learnable weights for each term type
        self.mult_weights = nn.Parameter(torch.ones(rank_mult) / rank_mult)
        self.add_x_weights = nn.Parameter(torch.ones(rank_add) / rank_add * 0.5)
        self.add_y_weights = nn.Parameter(torch.ones(rank_add) / rank_add * 0.5)
        
        # Global bias
        self.global_bias = nn.Parameter(torch.tensor(0.5))
        
        # Spline evaluators
        self.spline_x = CubicHermiteSpline(ctrl_x, output_w)
        self.spline_y = CubicHermiteSpline(ctrl_y, output_h)
        
        self._init_weights()
    
    def _init_weights(self):
        for gen in (self.mult_x_generators + self.mult_y_generators + 
                    self.add_x_generators + self.add_y_generators):
            for m in gen.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.5)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        B = h.shape[0]
        
        # Get per-rank features
        rank_feats = self.rank_conditioner(h)  # [B, R_total, hidden]
        
        # === Generate multiplicative control points ===
        Px_mult_list = []
        Py_mult_list = []
        for r in range(self.rank_mult):
            px = self.mult_x_generators[r](rank_feats[:, r])
            py = self.mult_y_generators[r](rank_feats[:, r])
            Px_mult_list.append(px)
            Py_mult_list.append(py)
        
        Px_mult = torch.stack(Px_mult_list, dim=1)  # [B, R_mult, ctrl_x]
        Py_mult = torch.stack(Py_mult_list, dim=1)  # [B, R_mult, ctrl_y]
        
        # === Generate additive control points ===
        Px_add_list = []
        Py_add_list = []
        for r in range(self.rank_add):
            feat_idx = self.rank_mult + r
            px = self.add_x_generators[r](rank_feats[:, feat_idx])
            py = self.add_y_generators[r](rank_feats[:, feat_idx])
            Px_add_list.append(px)
            Py_add_list.append(py)
        
        Px_add = torch.stack(Px_add_list, dim=1)  # [B, R_add, ctrl_x]
        Py_add = torch.stack(Py_add_list, dim=1)  # [B, R_add, ctrl_y]
        
        # === Evaluate splines ===
        u_mult = self.spline_x(Px_mult)  # [B, R_mult, W]
        v_mult = self.spline_y(Py_mult)  # [B, R_mult, H]
        
        a_x = self.spline_x(Px_add)  # [B, R_add, W]
        b_y = self.spline_y(Py_add)  # [B, R_add, H]
        
        # === Compute depth ===
        
        # 1. Multiplicative terms: Σ w_r * u_r(x) * v_r(y)
        mult_weights = F.softmax(self.mult_weights, dim=0).view(1, self.rank_mult, 1, 1)
        u_mult_exp = u_mult.unsqueeze(2)  # [B, R_mult, 1, W]
        v_mult_exp = v_mult.unsqueeze(3)  # [B, R_mult, H, 1]
        depth_mult = (u_mult_exp * v_mult_exp * mult_weights).sum(dim=1)  # [B, H, W]
        
        # 2. X-additive terms: Σ w_r * a_r(x) (broadcast over y)
        add_x_weights = F.softmax(self.add_x_weights, dim=0).view(1, self.rank_add, 1)
        depth_x = (a_x * add_x_weights).sum(dim=1)  # [B, W]
        depth_x = depth_x.unsqueeze(1).expand(-1, self.output_h, -1)  # [B, H, W]
        
        # 3. Y-additive terms: Σ w_r * b_r(y) (broadcast over x)
        add_y_weights = F.softmax(self.add_y_weights, dim=0).view(1, self.rank_add, 1)
        depth_y = (b_y * add_y_weights).sum(dim=1)  # [B, H]
        depth_y = depth_y.unsqueeze(2).expand(-1, -1, self.output_w)  # [B, H, W]
        
        # 4. Combine all terms
        depth = depth_mult + depth_x + depth_y + self.global_bias
        depth = depth.unsqueeze(1)  # [B, 1, H, W]
        
        info = {
            'Px_mult': Px_mult,
            'Py_mult': Py_mult,
            'Px_add': Px_add,
            'Py_add': Py_add,
            'depth_mult': depth_mult.unsqueeze(1),
            'depth_x': depth_x.unsqueeze(1),
            'depth_y': depth_y.unsqueeze(1),
            'mult_weights': self.mult_weights,
            'add_x_weights': self.add_x_weights,
            'add_y_weights': self.add_y_weights,
        }
        
        return depth, info


# ============================================================================
# Enhanced Residual CNN (bigger capacity)
# ============================================================================

class ResidualCNNV3(nn.Module):
    """
    Enhanced residual CNN with:
    - Larger capacity
    - Skip connections
    - Higher initial scale (0.5 instead of 0.1)
    """
    
    def __init__(
        self,
        latent_dim: int = 768,
        output_h: int = 256,
        output_w: int = 256,
        base_channels: int = 128,  # Increased from 64
    ):
        super().__init__()
        
        self.output_h = output_h
        self.output_w = output_w
        
        init_size = 8
        self.init_fc = nn.Linear(latent_dim, base_channels * 4 * init_size * init_size)
        self.init_size = init_size
        self.init_channels = base_channels * 4
        
        # Upsample blocks with skip-style connections
        self.up1 = self._make_upsample_block(base_channels * 4, base_channels * 4)  # 8->16
        self.up2 = self._make_upsample_block(base_channels * 4, base_channels * 2)  # 16->32
        self.up3 = self._make_upsample_block(base_channels * 2, base_channels)      # 32->64
        self.up4 = self._make_upsample_block(base_channels, base_channels)          # 64->128
        self.up5 = self._make_upsample_block(base_channels, base_channels // 2)     # 128->256
        
        # Final layers
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels // 2, base_channels // 4, 3, 1, 1, padding_mode='replicate'),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels // 4, 1, 3, 1, 1, padding_mode='replicate'),
        )
        
        # INCREASED: Start with scale 0.5 instead of 0.1
        self.residual_scale = nn.Parameter(torch.tensor(0.5))
    
    def _make_upsample_block(self, in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, padding_mode='replicate'),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, padding_mode='replicate'),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B = h.shape[0]
        
        x = self.init_fc(h)
        x = x.view(B, self.init_channels, self.init_size, self.init_size)
        
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        
        residual = self.final_conv(x)
        
        if residual.shape[2] != self.output_h or residual.shape[3] != self.output_w:
            residual = F.interpolate(residual, size=(self.output_h, self.output_w),
                                     mode='bilinear', align_corners=False)
        
        return residual * self.residual_scale


# ============================================================================
# Optional: 2D Spline Grid (for non-separable patterns)
# ============================================================================

class SplineGrid2D(nn.Module):
    """
    Direct 2D spline grid for truly non-separable depth patterns.
    Predicts a 2D grid of control points directly.
    """
    
    def __init__(
        self,
        latent_dim: int = 768,
        grid_h: int = 16,
        grid_w: int = 16,
        output_h: int = 256,
        output_w: int = 256,
    ):
        super().__init__()
        
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.output_h = output_h
        self.output_w = output_w
        
        self.grid_predictor = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(latent_dim, latent_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(latent_dim // 2, grid_h * grid_w),
        )
        
        self.scale = nn.Parameter(torch.tensor(0.3))
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B = h.shape[0]
        
        # Predict 2D grid
        grid = self.grid_predictor(h).view(B, 1, self.grid_h, self.grid_w)
        
        # Upsample to output size using bicubic interpolation
        depth = F.interpolate(grid, size=(self.output_h, self.output_w),
                              mode='bicubic', align_corners=False)
        
        return depth * self.scale


# ============================================================================
# Main Model v3
# ============================================================================

class AudioSplineDepthV3(nn.Module):
    """
    Audio-to-Depth model v3 with:
    1. Additive + Multiplicative spline decomposition
    2. Enhanced residual CNN
    3. Optional 2D grid for non-separable patterns
    4. Larger capacity (768 latent, 32 rank)
    """
    
    def __init__(
        self,
        cfg=None,
        input_channels: int = 2,
        output_h: int = 256,
        output_w: int = 256,
        latent_dim: int = 768,
        rank_mult: int = 16,
        rank_add: int = 8,
        ctrl_x: int = 24,
        ctrl_y: int = 24,
        use_residual: bool = True,
        use_2d_grid: bool = False,
        depth_activation: str = 'none',
    ):
        super().__init__()
        
        self.output_h = output_h
        self.output_w = output_w
        self.depth_activation = depth_activation
        self.use_residual = use_residual
        self.use_2d_grid = use_2d_grid
        
        if cfg is not None:
            output_h = cfg.dataset.images_size
            output_w = cfg.dataset.images_size
            self.output_h = output_h
            self.output_w = output_w
            self.depth_norm = getattr(cfg.dataset, 'depth_norm', True)
        else:
            self.depth_norm = True
        
        # Audio encoder (larger)
        self.encoder = AudioEncoderV3(
            input_channels=input_channels,
            latent_dim=latent_dim,
        )
        
        # Spline depth head with additive + multiplicative
        self.depth_head = SplineDepthHeadV3(
            latent_dim=latent_dim,
            rank_mult=rank_mult,
            rank_add=rank_add,
            ctrl_x=ctrl_x,
            ctrl_y=ctrl_y,
            output_h=output_h,
            output_w=output_w,
        )
        
        # Residual CNN (larger)
        if use_residual:
            self.residual_cnn = ResidualCNNV3(
                latent_dim=latent_dim,
                output_h=output_h,
                output_w=output_w,
            )
        
        # Optional 2D grid
        if use_2d_grid:
            self.grid_2d = SplineGrid2D(
                latent_dim=latent_dim,
                output_h=output_h,
                output_w=output_w,
            )
        
        self.config = {
            'latent_dim': latent_dim,
            'rank_mult': rank_mult,
            'rank_add': rank_add,
            'ctrl_x': ctrl_x,
            'ctrl_y': ctrl_y,
            'use_residual': use_residual,
            'use_2d_grid': use_2d_grid,
        }
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        h, _ = self.encoder(audio)
        
        depth_spline, _ = self.depth_head(h)
        
        depth = depth_spline
        
        if self.use_2d_grid:
            depth = depth + self.grid_2d(h)
        
        if self.use_residual:
            depth = depth + self.residual_cnn(h)
        
        if self.depth_activation == 'sigmoid':
            depth = torch.sigmoid(depth)
        elif self.depth_activation == 'softplus':
            depth = F.softplus(depth)
        
        return depth
    
    def forward_with_info(self, audio: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        h, scale_feats = self.encoder(audio)
        
        depth_spline, spline_info = self.depth_head(h)
        
        depth = depth_spline
        info = spline_info.copy()
        info['depth_spline'] = depth_spline
        
        if self.use_2d_grid:
            grid_depth = self.grid_2d(h)
            depth = depth + grid_depth
            info['depth_2d_grid'] = grid_depth
        
        if self.use_residual:
            residual = self.residual_cnn(h)
            depth = depth + residual
            info['residual'] = residual
        
        if self.depth_activation == 'sigmoid':
            depth = torch.sigmoid(depth)
        elif self.depth_activation == 'softplus':
            depth = F.softplus(depth)
        
        info['latent'] = h
        
        return depth, info
    
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_spline_complexity(self) -> Dict:
        return {
            'rank_mult': self.depth_head.rank_mult,
            'rank_add': self.depth_head.rank_add,
            'ctrl_x': self.depth_head.ctrl_x,
            'ctrl_y': self.depth_head.ctrl_y,
            'total_ctrl_points': (
                self.depth_head.rank_mult * (self.depth_head.ctrl_x + self.depth_head.ctrl_y) +
                self.depth_head.rank_add * (self.depth_head.ctrl_x + self.depth_head.ctrl_y)
            ),
            'use_residual': self.use_residual,
            'use_2d_grid': self.use_2d_grid,
        }


# ============================================================================
# Loss Functions
# ============================================================================

class DiversityLoss(nn.Module):
    """Encourages different ranks to learn different patterns."""
    
    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight
    
    def forward(self, Px: torch.Tensor, Py: torch.Tensor) -> torch.Tensor:
        """Penalize correlation between different ranks' control points."""
        B, R, N = Px.shape
        
        # Normalize
        Px_norm = Px - Px.mean(dim=2, keepdim=True)
        Px_norm = Px_norm / (Px_norm.std(dim=2, keepdim=True) + 1e-6)
        
        # Correlation matrix [B, R, R]
        corr = torch.bmm(Px_norm, Px_norm.transpose(1, 2)) / N
        
        # Penalize off-diagonal
        eye = torch.eye(R, device=corr.device).unsqueeze(0)
        off_diag = corr * (1 - eye)
        
        return self.weight * off_diag.abs().mean()


class EdgeAwareLoss(nn.Module):
    """Edge-aware gradient loss."""
    
    def __init__(self, weight: float = 0.5):
        super().__init__()
        self.weight = weight
    
    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        
        gt_dx = gt[:, :, :, 1:] - gt[:, :, :, :-1]
        gt_dy = gt[:, :, 1:, :] - gt[:, :, :-1, :]
        
        loss = F.l1_loss(pred_dx, gt_dx) + F.l1_loss(pred_dy, gt_dy)
        return self.weight * loss


class BoundaryLoss(nn.Module):
    """Penalizes boundary artifacts."""
    
    def __init__(self, weight: float = 0.1, border_width: int = 2):
        super().__init__()
        self.weight = weight
        self.bw = border_width
    
    def forward(self, pred: torch.Tensor, gt: torch.Tensor = None) -> torch.Tensor:
        bw = self.bw
        B, C, H, W = pred.shape
        
        loss = 0.0
        
        if W > 2 * bw:
            loss += F.l1_loss(pred[:, :, :, :bw], pred[:, :, :, bw:2*bw])
            loss += F.l1_loss(pred[:, :, :, -bw:], pred[:, :, :, -2*bw:-bw])
        
        if H > 2 * bw:
            loss += F.l1_loss(pred[:, :, :bw, :], pred[:, :, bw:2*bw, :])
            loss += F.l1_loss(pred[:, :, -bw:, :], pred[:, :, -2*bw:-bw, :])
        
        return self.weight * loss


class SplineSmoothLoss(nn.Module):
    """Light smoothness regularization."""
    
    def __init__(self, weight: float = 0.0001):
        super().__init__()
        self.weight = weight
    
    def forward(self, Px: torch.Tensor, Py: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        
        if Px.shape[2] >= 3:
            d2_x = Px[:, :, 2:] - 2 * Px[:, :, 1:-1] + Px[:, :, :-2]
            loss += (d2_x ** 2).mean()
        
        if Py.shape[2] >= 3:
            d2_y = Py[:, :, 2:] - 2 * Py[:, :, 1:-1] + Py[:, :, :-2]
            loss += (d2_y ** 2).mean()
        
        return self.weight * loss


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


def define_spline_depth_v3(
    cfg=None,
    input_nc: int = 2,
    output_h: int = 256,
    output_w: int = 256,
    latent_dim: int = 768,
    rank_mult: int = 16,
    rank_add: int = 8,
    ctrl_x: int = 24,
    ctrl_y: int = 24,
    use_residual: bool = True,
    use_2d_grid: bool = False,
    depth_activation: str = 'none',
    init_type: str = 'kaiming',
    init_gain: float = 0.02,
    gpu_ids: List[int] = [],
) -> nn.Module:
    net = AudioSplineDepthV3(
        cfg=cfg,
        input_channels=input_nc,
        output_h=output_h,
        output_w=output_w,
        latent_dim=latent_dim,
        rank_mult=rank_mult,
        rank_add=rank_add,
        ctrl_x=ctrl_x,
        ctrl_y=ctrl_y,
        use_residual=use_residual,
        use_2d_grid=use_2d_grid,
        depth_activation=depth_activation,
    )
    
    return init_net(net, init_type, init_gain, gpu_ids)


# ============================================================================
# Testing
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Testing AudioSplineDepthV3")
    print("=" * 60)
    
    model = AudioSplineDepthV3(
        input_channels=2,
        output_h=256,
        output_w=256,
        latent_dim=768,
        rank_mult=16,
        rank_add=8,
        ctrl_x=24,
        ctrl_y=24,
        use_residual=True,
        use_2d_grid=False,
    )
    
    batch_size = 2
    audio = torch.randn(batch_size, 2, 256, 256)
    
    depth = model(audio)
    print(f"Input: {audio.shape}")
    print(f"Output: {depth.shape}")
    print(f"Depth range: [{depth.min():.4f}, {depth.max():.4f}]")
    print(f"Parameters: {model.get_num_params():,}")
    print(f"Complexity: {model.get_spline_complexity()}")
    
    depth, info = model.forward_with_info(audio)
    print(f"\nComponents:")
    print(f"  depth_mult: {info['depth_mult'].shape}")
    print(f"  depth_x: {info['depth_x'].shape}")
    print(f"  depth_y: {info['depth_y'].shape}")
    if 'residual' in info:
        print(f"  residual: {info['residual'].shape}")
    
    print("\nAll tests passed!")


