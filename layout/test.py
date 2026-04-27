import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
import datetime
import random
import torch
import cv2
import os


from model import Echo_Layout
from loss import LayoutLoss

from Dataloaders.communal.utils.conversion import depth2xyz, uv2xyz, get_u, depth2uv, xyz2uv, uv2pixel
from Dataloaders.communal.utils.boundary import corners2boundaries, layout2depth, draw_boundaries

from torch.utils.data import DataLoader, Dataset
from Dataloaders.dataset_matterport_layout import MP3D_Layout_Dataset
from metric import calc_accuracy
from tqdm import tqdm


def visualize_2d(img, depth, ratio, pred_depth, pred_ratio, show_depth=True, show_floorplan=True, show=False, save_path=None):
    #dt_np = tensor2np_d(dt)
    dt_depth = pred_depth[0]
    dt_xyz = depth2xyz(np.abs(dt_depth))
    dt_ratio = pred_ratio[0][0]
    dt_boundaries = corners2boundaries(dt_ratio, corners_xyz=dt_xyz, step=None, visible=False, length=img.shape[1])
    vis_img = draw_boundaries(img[:,:,::-1], boundary_list=dt_boundaries, boundary_color=[0, 1, 0])

    gt_depth = depth[0]
    gt_xyz = depth2xyz(np.abs(gt_depth))
    gt_ratio = ratio[0][0]
    gt_boundaries = corners2boundaries(gt_ratio, corners_xyz=gt_xyz, step=None, visible=False, length=vis_img.shape[1])
    vis_img = draw_boundaries(vis_img, boundary_list=gt_boundaries, boundary_color=[0, 0, 1])
    return vis_img


def eval(model, test_loader, args):
    count = 0
    performance = {
        'visible_2d': [],
        'visible_3d': [],
        'height': []}
    
    with torch.no_grad():
        model.eval()
        for batch, (rgb, depth, label) in tqdm(enumerate(test_loader)):
            rgb = rgb.permute(0,3,1,2).cuda()
            depth = depth.unsqueeze(1).cuda()
            ratio = label[:, :, 0].cuda()
            depth_1d = label[:, :, 1:].cuda()

            pred_depth, pred_ratio = Net(depth)

            # Draw GT & pred boundaries
            layout_2d = visualize_2d((rgb.detach().cpu().numpy())[0].transpose(1,2,0), depth_1d.detach().cpu().numpy(), ratio.detach().cpu().numpy(), pred_depth.detach().cpu().numpy(), pred_ratio.detach().cpu().numpy(), show_depth=True, show_floorplan=True, show=False)

            visb_iou, iou_height = calc_accuracy(pred_depth.detach().cpu().numpy(), pred_ratio.detach().cpu().numpy(), depth_1d.detach().cpu().numpy(), ratio.detach().cpu().numpy(), h=256)
            performance['visible_2d'].append(visb_iou[0])
            performance['visible_3d'].append(visb_iou[1])
            performance['height'].append(iou_height)
            
            cv2.imwrite("./results/v2/" + str(count).zfill(4) + "_layout2d.png", (layout_2d*255).astype(np.uint8))

            count += 1

        visible_2d = np.array(performance['visible_2d']).mean()
        visible_3d = np.array(performance['visible_3d']).mean()
        height = np.array(performance['height']).mean()

    print("visible_2d:", visible_2d)
    print("visible_3d:", visible_3d)
    print("height", height)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='matterport3D_layout')
    parser.add_argument('--rootpath', type=str, default='/home/jongsung/matterport3d/')
    parser.add_argument('--trainfile', type=str, default='./filenames/layout/matterport3d_layout_train.txt')
    parser.add_argument('--valfile', type=str, default='./filenames/layout/matterport3d_layout_val.txt')

    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/Layout_Best_IoU_v2.pth')
    parser.add_argument('--save_path', type=str, default='./results/v2/')
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('-g', '--gpu_id', default=7, type=int, help='gpu id setting')
    args = parser.parse_args()
    
    test_loader = torch.utils.data.DataLoader(dataset = MP3D_Layout_Dataset(
                                                root_dir=args.rootpath,
                                                mode='test',
                                                shape=[512, 1024],
                                                max_wall_num=0,
                                                aug=None,
                                                camera_height=1.6,
                                                logger=None,
                                                for_test_index=None,
                                                keys=[]), batch_size=1, shuffle=False, num_workers=16, drop_last = False, pin_memory=True)

    # Load trained model parameters
    ckpt = torch.load(args.checkpoint_path)
    Net = Echo_Layout().cuda()
    Net.load_state_dict(ckpt)
    Net = Net.eval()

    # Test
    eval(Net, test_loader, args)