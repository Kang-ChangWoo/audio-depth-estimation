"""
@date: 2021/6/25
@description:
"""
import os
import json
import numpy as np
import torch

from .communal.read import read_image, read_depth,read_label
from .communal.base_dataset import BaseDataset
from .communal.utils.logger import get_logger


class MP3D_Layout_Dataset(BaseDataset):
    def __init__(self, root_dir, mode, shape=None, max_wall_num=0, aug=None, camera_height=1.6, logger=None,
                 split_list=None, patch_num=256, keys=None, for_test_index=None):
        super().__init__(mode, shape, max_wall_num, aug, camera_height, patch_num, keys)

        if logger is None:
            logger = get_logger()
        self.root_dir = root_dir

        split_dir = os.path.join('/root/storage/implementation/shared_audio/baseline/layout/Dataloaders/split/')

        if split_list is None:
            with open(split_dir+f"{mode}.txt", 'r') as f:
                split_list = [x.rstrip().split() for x in f]

        split_list.sort()
        if for_test_index is not None:
            split_list = split_list[:for_test_index]

        self.data = []
        invalid_num = 0
        for name in split_list:
            img_path = name[0]
            depth_path = name[1]
            label_path = name[2]

            if not os.path.exists(depth_path):
                continue
            else:
                self.data.append([img_path, depth_path, label_path])
                

    def __getitem__(self, idx):
        rgb_path, depth_path, label_path = self.data[idx]
        image = read_image(rgb_path, self.shape)[:,:,:3]
        depth = np.load(depth_path)
        label = np.load(label_path)

        return image, depth, label