"""
Knowledge Distillation Loss for RGB → Audio Transfer

Loss Components:
1. Task Loss: Audio's depth prediction accuracy
2. Response Distillation: Match RGB's final predictions (soft targets)
3. Feature Distillation: Match RGB's intermediate features
4. Bin Distribution Distillation: Match RGB's bin predictions
5. Residual Sparsity: Encourage small residuals

Temperature scaling for soft targets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss
    
    Args:
        lambda_task: Weight for task loss (audio depth vs GT)
        lambda_response: Weight for response distillation (audio vs RGB predictions)
        lambda_feature: Weight for feature distillation (audio vs RGB features)
        lambda_bin: Weight for bin distribution matching
        lambda_sparse: Weight for residual sparsity
        temperature: Temperature for soft targets (higher = softer)
    """
    
    def __init__(
        self,
        lambda_task=2.0,  # HIGH - need to learn from GT!
        lambda_response=0.3,  # Lower - don't over-rely on teacher
        lambda_feature=0.2,  # Lower - feature matching is secondary
        lambda_bin=0.05,  # Much lower - KL loss is naturally large
        lambda_sparse=0.1,
        temperature=4.0
    ):
        super().__init__()
        self.lambda_task = lambda_task
        self.lambda_response = lambda_response
        self.lambda_feature = lambda_feature
        self.lambda_bin = lambda_bin
        self.lambda_sparse = lambda_sparse
        self.temperature = temperature
    
    def task_loss(self, audio_depth, gt_depth, valid_mask):
        """
        Task Loss: Audio prediction vs Ground Truth
        
        Standard depth estimation loss (L1)
        """
        if valid_mask is not None:
            loss = F.l1_loss(audio_depth[valid_mask], gt_depth[valid_mask])
        else:
            loss = F.l1_loss(audio_depth, gt_depth)
        return loss
    
    def response_distillation_loss(self, audio_depth, rgb_depth, valid_mask=None):
        """
        Response Distillation: Match teacher's final predictions
        
        Uses MSE to match soft targets (teacher's predictions)
        """
        if valid_mask is not None:
            loss = F.mse_loss(audio_depth[valid_mask], rgb_depth[valid_mask].detach())
        else:
            loss = F.mse_loss(audio_depth, rgb_depth.detach())
        return loss
    
    def feature_distillation_loss(self, audio_features, rgb_features):
        """
        Feature Distillation: Match teacher's intermediate features
        
        Match features at each encoder level using L2 distance
        """
        total_loss = 0.0
        count = 0
        
        for level in ['x1', 'x2', 'x3', 'x4', 'x5']:
            if level in audio_features and level in rgb_features:
                audio_feat = audio_features[level]
                rgb_feat = rgb_features[level].detach()
                
                # Normalize features to unit norm for stable training
                audio_feat_norm = F.normalize(audio_feat.flatten(2), dim=2)
                rgb_feat_norm = F.normalize(rgb_feat.flatten(2), dim=2)
                
                # Cosine distance (1 - cosine similarity)
                cos_sim = (audio_feat_norm * rgb_feat_norm).sum(dim=2).mean()
                level_loss = 1 - cos_sim
                
                total_loss += level_loss
                count += 1
        
        return total_loss / count if count > 0 else torch.tensor(0.0)
    
    def bin_distribution_loss(self, audio_logits, rgb_logits, temperature=None):
        """
        Bin Distribution Distillation: Match teacher's bin predictions
        
        Uses KL divergence with temperature scaling for soft targets
        
        CRITICAL FIX: Spatial average first to avoid huge loss values
        bin_logits: [B, n_bins, H, W] → [B, n_bins] before KL divergence
        """
        if temperature is None:
            temperature = self.temperature
        
        # Spatial average to get per-image bin distribution
        # [B, n_bins, H, W] → [B, n_bins]
        audio_logits_avg = audio_logits.mean(dim=(2, 3))
        rgb_logits_avg = rgb_logits.mean(dim=(2, 3))
        
        # Apply temperature scaling
        audio_soft = F.log_softmax(audio_logits_avg / temperature, dim=1)
        rgb_soft = F.softmax(rgb_logits_avg.detach() / temperature, dim=1)
        
        # KL divergence
        kl_loss = F.kl_div(audio_soft, rgb_soft, reduction='batchmean')
        
        # NO temperature^2 scaling - already huge without it!
        return kl_loss
    
    def bin_centers_loss(self, audio_bins, rgb_bins):
        """
        Bin Centers Matching: Encourage similar bin distributions
        
        Match the adaptive bin centers predicted by student and teacher
        """
        return F.mse_loss(audio_bins, rgb_bins.detach())
    
    def residual_sparsity_loss(self, residual, valid_mask=None):
        """
        Residual Sparsity: Encourage small residuals
        
        We want base depth to do most of the work
        """
        if valid_mask is not None:
            loss = torch.abs(residual[valid_mask]).mean()
        else:
            loss = torch.abs(residual).mean()
        return loss
    
    def forward(self, output, gt_depth, valid_mask=None):
        """
        Compute total distillation loss
        
        Args:
            output: Dictionary from model forward pass
                - output['audio']: Audio predictions
                - output['rgb']: RGB predictions (teacher)
            gt_depth: [B, 1, H, W] - Ground truth depth
            valid_mask: [B, 1, H, W] - Valid depth mask
        
        Returns:
            total_loss: Scalar tensor
            loss_dict: Dictionary with individual loss values
        """
        audio_out = output['audio']
        rgb_out = output['rgb']
        
        # ==========================================
        # 1. Task Loss (always computed)
        # ==========================================
        loss_task = self.task_loss(
            audio_out['final_depth'],
            gt_depth,
            valid_mask
        )
        
        # ==========================================
        # 2. Distillation Losses (only if RGB available)
        # ==========================================
        if rgb_out is not None:
            # Response distillation
            loss_response = self.response_distillation_loss(
                audio_out['final_depth'],
                rgb_out['final_depth'],
                valid_mask
            )
            
            # Feature distillation
            loss_feature = self.feature_distillation_loss(
                audio_out['features'],
                rgb_out['features']
            )
            
            # Bin distribution distillation
            loss_bin = self.bin_distribution_loss(
                audio_out['bin_logits'],
                rgb_out['bin_logits']
            )
            
            # Bin centers matching
            loss_bin_centers = self.bin_centers_loss(
                audio_out['bin_centers'],
                rgb_out['bin_centers']
            )
        else:
            # Inference mode: no distillation
            loss_response = torch.tensor(0.0)
            loss_feature = torch.tensor(0.0)
            loss_bin = torch.tensor(0.0)
            loss_bin_centers = torch.tensor(0.0)
        
        # ==========================================
        # 3. Residual Sparsity
        # ==========================================
        loss_sparse = self.residual_sparsity_loss(
            audio_out['residual'],
            valid_mask
        )
        
        # ==========================================
        # Total Loss
        # ==========================================
        total_loss = (
            self.lambda_task * loss_task +
            self.lambda_response * loss_response +
            self.lambda_feature * loss_feature +
            self.lambda_bin * (loss_bin + loss_bin_centers) +
            self.lambda_sparse * loss_sparse
        )
        
        loss_dict = {
            'task': loss_task.item(),
            'response': loss_response.item() if isinstance(loss_response, torch.Tensor) else loss_response,
            'feature': loss_feature.item() if isinstance(loss_feature, torch.Tensor) else loss_feature,
            'bin': loss_bin.item() if isinstance(loss_bin, torch.Tensor) else loss_bin,
            'bin_centers': loss_bin_centers.item() if isinstance(loss_bin_centers, torch.Tensor) else loss_bin_centers,
            'sparse': loss_sparse.item(),
            'total': total_loss.item()
        }
        
        return total_loss, loss_dict


class AdaptiveDistillationLoss(nn.Module):
    """
    Adaptive Distillation Loss with curriculum learning
    
    Gradually shifts from relying on teacher to independent learning
    
    Phase 1 (epochs 0-20): Heavy distillation (learn from teacher)
    Phase 2 (epochs 20-80): Balanced (distillation + task loss)
    Phase 3 (epochs 80+): Light distillation (mostly task loss)
    """
    
    def __init__(
        self,
        max_epochs=200,
        temperature=4.0,
        lambda_sparse=0.1
    ):
        super().__init__()
        self.max_epochs = max_epochs
        self.temperature = temperature
        self.lambda_sparse = lambda_sparse
        self.current_epoch = 0
    
    def set_epoch(self, epoch):
        """Update current epoch for adaptive weighting"""
        self.current_epoch = epoch
    
    def get_adaptive_weights(self):
        """
        Compute adaptive loss weights based on training progress
        
        Returns:
            Dictionary with adaptive weights
        """
        progress = min(1.0, self.current_epoch / self.max_epochs)
        
        # Task loss: HIGH from start (need to learn from GT!)
        lambda_task = 2.0 + progress  # Start at 2.0, increase to 3.0
        
        # Response distillation: start LOW, gradually increase
        # (early: RGB is random, don't follow it too much)
        if progress < 0.1:
            lambda_response = 0.1  # Very low for first 20 epochs
        else:
            lambda_response = 0.1 + 0.4 * (progress - 0.1) / 0.9  # Increase to 0.5
        
        # Feature distillation: low initially, peak in middle
        if progress < 0.2:
            lambda_feature = 0.05  # Very low initially
        elif progress < 0.5:
            lambda_feature = 0.05 + 0.25 * (progress - 0.2) / 0.3  # Increase to 0.3
        else:
            lambda_feature = 0.3 - 0.1 * (progress - 0.5) / 0.5  # Decrease to 0.2
        
        # Bin distillation: MUCH lower weight (KL loss is naturally large)
        lambda_bin = 0.05 - 0.03 * progress  # Start at 0.05, decrease to 0.02
        
        return {
            'task': lambda_task,
            'response': lambda_response,
            'feature': lambda_feature,
            'bin': lambda_bin,
            'sparse': self.lambda_sparse
        }
    
    def forward(self, output, gt_depth, valid_mask=None):
        """
        Compute adaptive distillation loss
        
        Args:
            output: Dictionary from model forward pass
            gt_depth: [B, 1, H, W] - Ground truth depth
            valid_mask: [B, 1, H, W] - Valid depth mask
        
        Returns:
            total_loss: Scalar tensor
            loss_dict: Dictionary with individual loss values
        """
        # Get adaptive weights
        weights = self.get_adaptive_weights()
        
        # Create standard distillation loss with adaptive weights
        criterion = DistillationLoss(
            lambda_task=weights['task'],
            lambda_response=weights['response'],
            lambda_feature=weights['feature'],
            lambda_bin=weights['bin'],
            lambda_sparse=weights['sparse'],
            temperature=self.temperature
        )
        
        total_loss, loss_dict = criterion(output, gt_depth, valid_mask)
        
        # Add weight info to loss dict
        loss_dict['weights'] = weights
        
        return total_loss, loss_dict


# ==========================================
# Test Code
# ==========================================
if __name__ == '__main__':
    print("Testing Distillation Loss")
    print("=" * 60)
    
    # Create dummy data
    B, H, W = 4, 256, 256
    n_bins = 128
    
    # Dummy output from model
    output = {
        'audio': {
            'features': {
                'x1': torch.randn(B, 64, H, W),
                'x2': torch.randn(B, 128, H//2, W//2),
                'x3': torch.randn(B, 256, H//4, W//4),
                'x4': torch.randn(B, 512, H//8, W//8),
                'x5': torch.randn(B, 512, H//16, W//16),
            },
            'bin_centers': torch.randn(B, n_bins),
            'bin_logits': torch.randn(B, n_bins, H, W),
            'base_depth': torch.randn(B, 1, H, W),
            'residual': torch.randn(B, 1, H, W) * 0.1,
            'final_depth': torch.randn(B, 1, H, W),
        },
        'rgb': {
            'features': {
                'x1': torch.randn(B, 64, H, W),
                'x2': torch.randn(B, 128, H//2, W//2),
                'x3': torch.randn(B, 256, H//4, W//4),
                'x4': torch.randn(B, 512, H//8, W//8),
                'x5': torch.randn(B, 512, H//16, W//16),
            },
            'bin_centers': torch.randn(B, n_bins),
            'bin_logits': torch.randn(B, n_bins, H, W),
            'final_depth': torch.randn(B, 1, H, W),
        }
    }
    
    gt_depth = torch.randn(B, 1, H, W).abs() * 10
    valid_mask = torch.ones(B, 1, H, W).bool()
    
    # Test standard loss
    print("\n=== Standard Distillation Loss ===")
    criterion = DistillationLoss()
    loss, loss_dict = criterion(output, gt_depth, valid_mask)
    
    print(f"Total loss: {loss.item():.4f}")
    for key, value in loss_dict.items():
        if key != 'weights':
            print(f"  {key:15s}: {value:.4f}")
    
    # Test adaptive loss
    print("\n=== Adaptive Distillation Loss ===")
    adaptive_criterion = AdaptiveDistillationLoss(max_epochs=100)
    
    for epoch in [0, 25, 50, 75, 100]:
        adaptive_criterion.set_epoch(epoch)
        loss, loss_dict = adaptive_criterion(output, gt_depth, valid_mask)
        
        print(f"\nEpoch {epoch}:")
        print(f"  Total loss: {loss.item():.4f}")
        print(f"  Weights: task={loss_dict['weights']['task']:.2f}, "
              f"response={loss_dict['weights']['response']:.2f}, "
              f"feature={loss_dict['weights']['feature']:.2f}")
    
    print("\n✅ Loss test passed!")


