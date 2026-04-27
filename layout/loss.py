import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch

from Dataloaders.communal.utils.conversion import depth2xyz, xyz2lonlat


class LayoutLoss(nn.Module):
    def __init__(self):
        super(LayoutLoss, self).__init__()
        self.boundary_loss = nn.L1Loss()
        self.depth_loss = nn.L1Loss()

    def forward(self, pred_depth, pred_ratio, depth, ratio):
        gt_floor_xyz = depth2xyz(depth)
        gt_ceil_xyz = gt_floor_xyz.clone()
        gt_ceil_xyz[..., 1] = -ratio

        gt_floor_boundary = xyz2lonlat(gt_floor_xyz)[..., -1:]
        gt_ceil_boundary = xyz2lonlat(gt_ceil_xyz)[..., -1:]
        gt_boundary = torch.cat([gt_floor_boundary, gt_ceil_boundary], dim=-1).permute(0, 2, 1)

        pred_xyz = depth2xyz(depth)
        pred_floor_boundary = xyz2lonlat(pred_xyz)
        pred_floor_boundary = pred_floor_boundary[..., -1:]

        pred_xyz_y = depth2xyz(pred_depth, plan_y=-pred_ratio)
        pred_ceil_boundary = xyz2lonlat(pred_xyz_y)
        pred_ceil_boundary = pred_ceil_boundary[..., -1:]
        pred_boundary = torch.cat([pred_floor_boundary, pred_ceil_boundary], dim=-1).permute(0, 2, 1)

        boundary_loss = self.boundary_loss(gt_boundary, pred_boundary)
        depth_loss = self.depth_loss(depth, pred_depth)
        
        loss = boundary_loss + depth_loss
        return loss