import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
import datetime
import random
import torch
import os


from model import Echo_Layout
from loss import LayoutLoss

from torch.utils.data import DataLoader, Dataset
from Dataloaders.dataset_matterport_layout import MP3D_Layout_Dataset
from metric import calc_accuracy
from tqdm import tqdm


try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            pass

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            pass



def main(args):
    input_dir = args.rootpath
    train_file = args.trainfile
    val_file = args.valfile
    batch_size = args.bs
    epochs = args.epoch
    
    train_loader = torch.utils.data.DataLoader(dataset = MP3D_Layout_Dataset(
                                                root_dir=input_dir,
                                                mode='train',
                                                shape=[512, 1024],
                                                max_wall_num=0,
                                                aug=None,
                                                camera_height=1.6,
                                                logger=None,
                                                for_test_index=None,
                                                keys=[]), batch_size=batch_size, shuffle=True, num_workers=16, drop_last = True, pin_memory=True )

    val_loader = torch.utils.data.DataLoader(dataset = MP3D_Layout_Dataset(
                                                root_dir=input_dir,
                                                mode='val',
                                                shape=[512, 1024],
                                                max_wall_num=0,
                                                aug=None,
                                                camera_height=1.6,
                                                logger=None,
                                                for_test_index=None,
                                                keys=[]), batch_size=1, shuffle=False, num_workers=16, drop_last = False, pin_memory=True)


    Net = Echo_Layout().cuda()

    optimizer = optim.AdamW(list(Net.parameters()), lr=1e-5, betas=(0.9, 0.999), weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6, last_epoch=-1)
    scaler = GradScaler()
    
    criterion_layout = LayoutLoss()
    best_iou = 0

    for epoch in range(epochs):
        Net.train()
        for batch, (rgb, depth, label) in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()

            depth = depth.unsqueeze(1).cuda()
            ratio = label[:, :, 0].cuda()
            depth_1d = label[:, :, 1:].cuda()

            pred_depth_1d, pred_ratio = Net(depth)
            loss = criterion_layout(pred_depth_1d, pred_ratio, depth_1d, ratio)
            
        
            if batch % 200 == 0 and batch > 0:
                print('[Epoch %d--Iter %d]Total loss %.4f' % (epoch, batch, loss))
        

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(Net.parameters(), 1.0)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

        ## Validation per 1 epoch 
        Net.eval()
        visible_2d, visible_3d, height = Validate(Net, val_loader)

        # Save the best model
        if visible_2d > best_iou:
            torch.save(Net.state_dict(), './checkpoints/Layout_Best_IoU_v2.pth')
            best_iou = visible_2d



def Validate(Net, val_loader):
    layout_performance = {
        'visible_2d': [],
        'visible_3d': [],
        'height': []}

    with torch.no_grad():
        for batch, (rgb, depth, label) in tqdm(enumerate(val_loader)):
            
            rgb = rgb.cuda()
            depth = depth.unsqueeze(1).cuda()
            ratio = label[:, :, 0].cuda()
            depth_1d = label[:, :, 1:].cuda()
    
            # corners = gt['corners']

            pred_depth_1d, pred_ratio = Net(depth)

            visb_iou, iou_height = calc_accuracy(pred_depth_1d.detach().cpu().numpy(), pred_ratio.detach().cpu().numpy(), depth_1d.detach().cpu().numpy(), ratio.detach().cpu().numpy(), h=256)

            layout_performance['visible_2d'].append(visb_iou[0])
            layout_performance['visible_3d'].append(visb_iou[1])
            layout_performance['height'].append(iou_height)

        visible_2d = np.array(layout_performance['visible_2d']).mean()
        visible_3d = np.array(layout_performance['visible_3d']).mean()
        height = np.array(layout_performance['height']).mean()
    
    print("---Layout validation score---")
    print("visible_2d:", visible_2d)
    print("visible_3d:", visible_3d)
    print("height", height)

    return visible_2d, visible_3d, height


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='matterport3D_layout')
    parser.add_argument('--rootpath', type=str, default='/home/jongsung/matterport3d/')
    parser.add_argument('--trainfile', type=str, default='./filenames/layout/matterport3d_layout_train.txt')
    parser.add_argument('--valfile', type=str, default='./filenames/layout/matterport3d_layout_val.txt')

    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=1000)    # 200 for Zind & 1000 for the others
    parser.add_argument('-g', '--gpu_id', default=7, type=int, help='gpu id setting')
    args = parser.parse_args()
    
    # GPU ID setting 
    if args.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    
    # Fix seed 
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    main(args)