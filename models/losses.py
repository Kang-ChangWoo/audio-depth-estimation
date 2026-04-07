"""Loss functions for depth estimation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SILogLoss(nn.Module):
    """Scale-Invariant Logarithmic Loss."""

    def __init__(self, variance_weight=0.5, epsilon=1e-6):
        super().__init__()
        self.variance_weight = variance_weight
        self.epsilon = epsilon

    def forward(self, pred, target):
        mask = target > 0
        pred, target = pred[mask], target[mask]
        if pred.numel() == 0:
            return torch.tensor(0.0, device=pred.device)
        pred = torch.clamp(pred, min=self.epsilon)
        target = torch.clamp(target, min=self.epsilon)
        log_diff = torch.log(pred) - torch.log(target)
        return torch.mean(log_diff ** 2) - self.variance_weight * (torch.mean(log_diff) ** 2)


class BerHuLoss(nn.Module):
    """Reverse Huber (BerHu) loss for depth estimation."""

    def forward(self, pred, gt):
        mask = gt > 0
        pred, gt = pred[mask], gt[mask]
        if pred.numel() == 0:
            return torch.tensor(0.0, device=pred.device)
        diff = torch.abs(pred - gt)
        c = 0.2 * diff.max().detach()
        l1 = diff[diff <= c]
        l2 = diff[diff > c]
        return (l1.sum() + ((l2 ** 2 + c ** 2) / (2 * c)).sum()) / pred.numel()


class DepthLoss(nn.Module):
    """Combined depth loss. Default: BerHu + SILog."""

    def __init__(self, use_berhu=True, use_silog=True,
                 w_berhu=1.0, w_silog=0.5):
        super().__init__()
        self.use_berhu = use_berhu
        self.use_silog = use_silog
        if use_berhu: self.berhu = BerHuLoss();  self.w_berhu = w_berhu
        if use_silog: self.silog = SILogLoss();   self.w_silog = w_silog

    def forward(self, pred, gt):
        loss = torch.tensor(0.0, device=pred.device)
        if self.use_berhu: loss = loss + self.w_berhu * self.berhu(pred, gt)
        if self.use_silog: loss = loss + self.w_silog * self.silog(pred, gt)
        return loss


class FOAGuidedLoss(nn.Module):
    """FOA prediction loss: L1 + optional cosine similarity."""

    def __init__(self, use_cosine=True, cosine_weight=0.1):
        super().__init__()
        self.use_cosine = use_cosine
        self.cosine_weight = cosine_weight

    def forward(self, pred_foa, gt_foa):
        l1 = F.l1_loss(pred_foa, gt_foa)
        if self.use_cosine:
            cos = 1.0 - F.cosine_similarity(pred_foa, gt_foa, dim=-1).mean()
            return l1 + self.cosine_weight * cos
        return l1


class SHHistogramAlignmentLoss(nn.Module):
    """Alignment loss between SH energy reconstruction and depth structure."""

    def __init__(self, map_cosine_weight=0.5, coeff_cosine_weight=0.2):
        super().__init__()
        self.map_cosine_weight = map_cosine_weight
        self.coeff_cosine_weight = coeff_cosine_weight

    def _normalize_map(self, x, eps=1e-6):
        B = x.shape[0]
        x_flat = x.view(B, -1)
        x_min = x_flat.min(dim=1, keepdim=True).values
        x_max = x_flat.max(dim=1, keepdim=True).values
        return ((x_flat - x_min) / (x_max - x_min + eps)).view_as(x)

    def forward(self, energy_recon_aligned, depth_sh,
                sh_aligned=None, depth_sh_coeffs=None):
        e_norm = self._normalize_map(energy_recon_aligned)
        d_norm = self._normalize_map(depth_sh)
        loss = F.l1_loss(e_norm, d_norm)

        B = e_norm.shape[0]
        e_flat = e_norm.view(B, -1)
        d_flat = d_norm.view(B, -1)
        loss = loss + self.map_cosine_weight * (
            1.0 - F.cosine_similarity(e_flat, d_flat, dim=-1).mean())

        if sh_aligned is not None and depth_sh_coeffs is not None:
            loss = loss + self.coeff_cosine_weight * (
                1.0 - F.cosine_similarity(sh_aligned, depth_sh_coeffs, dim=-1).mean())
        return loss


class AudioDepthFOALoss(nn.Module):
    """Combined loss for FOA model: depth + FOA + histogram alignment."""

    def __init__(self, depth_criterion, foa_criterion=None,
                 depth_weight=1.0, foa_weight=0.1,
                 hist_criterion=None, hist_weight=0.1, latent_reg_weight=0.0):
        super().__init__()
        self.depth_criterion = depth_criterion
        self.foa_criterion = foa_criterion if foa_criterion is not None else FOAGuidedLoss()
        self.hist_criterion = hist_criterion
        self.depth_weight = depth_weight
        self.foa_weight = foa_weight
        self.hist_weight = hist_weight
        self.latent_reg_weight = latent_reg_weight

    def forward(self, outputs, gt_depth, gt_foa,
                gt_depth_sh=None, gt_depth_sh_coeffs=None):
        pred_depth = outputs["pred_depth"]
        pred_foa = outputs["pred_foa"]
        foa_latent = outputs["foa_latent"]

        depth_loss = self.depth_criterion(pred_depth, gt_depth)
        foa_loss = self.foa_criterion(pred_foa, gt_foa)

        hist_loss = torch.tensor(0.0, device=pred_depth.device)
        if (self.hist_criterion is not None and self.hist_weight > 0
                and "energy_recon_aligned" in outputs):
            energy_recon = outputs["energy_recon_aligned"]
            depth_sh = outputs["depth_sh"]
            sh_aligned = outputs.get("sh_aligned")
            depth_sh_coeffs = outputs.get("depth_sh_coeffs")

            if gt_depth_sh is not None:
                ambi_guide = self.hist_criterion(
                    energy_recon, gt_depth_sh, sh_aligned, gt_depth_sh_coeffs)
                depth_follow = self.hist_criterion(
                    energy_recon.detach(), depth_sh,
                    sh_aligned.detach() if sh_aligned is not None else None,
                    depth_sh_coeffs)
                hist_loss = ambi_guide + depth_follow
            else:
                hist_loss = self.hist_criterion(
                    energy_recon, depth_sh, sh_aligned, depth_sh_coeffs)

        latent_reg = torch.tensor(0.0, device=pred_depth.device)
        if self.latent_reg_weight > 0:
            latent_reg = (foa_latent ** 2).mean()

        total = (self.depth_weight * depth_loss
                 + self.foa_weight * foa_loss
                 + self.hist_weight * hist_loss
                 + self.latent_reg_weight * latent_reg)

        return {
            "total": total, "depth": depth_loss,
            "foa": foa_loss, "hist": hist_loss,
        }
