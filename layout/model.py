import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Any, Dict, Optional, Tuple


class Echo_Layout(nn.Module):
    def __init__(self):
        super(Echo_Layout, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        
        self.depth_head = nn.Linear(128, 1)
        self.depth_mlp = nn.Linear(512, 256)

        self.ratio_mlp = nn.Linear(128, 1)
        self.ratio_head = nn.Linear(512, 1)

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)



    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))

        x = x.flatten(2)
        
        depth_1d = self.depth_mlp(x.clone())
        depth_1d = depth_1d.permute(0,2,1)
        depth_1d = self.depth_head(depth_1d)

        ratio = self.ratio_mlp(x.clone().permute(0,2,1))
        ratio = ratio.squeeze(-1)
        ratio = self.ratio_head(ratio)

        depth_1d = torch.sigmoid(depth_1d) * 10.0
        ratio = torch.sigmoid(ratio)

        return depth_1d.permute(0,2,1), ratio