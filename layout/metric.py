import numpy as np
import cv2
import scipy

from shapely.geometry import Polygon
from Dataloaders.communal.utils.conversion import depth2xyz, uv2xyz, get_u, depth2uv, xyz2uv, uv2pixel
from Dataloaders.communal.utils.boundary import corners2boundaries, layout2depth


def calc_inter_area(dt_xz, gt_xz):
    """
    :param dt_xz: Prediction boundaries can also be corners, format: [[x1, z1], [x2, z2], ...]
    :param gt_xz: Ground truth boundaries can also be corners, format: [[x1, z1], [x2, z2], ...]
    :return:
    """
    dt_polygon = Polygon(dt_xz)
    gt_polygon = Polygon(gt_xz)

    dt_area = dt_polygon.area
    gt_area = gt_polygon.area
    inter_area = dt_polygon.intersection(gt_polygon).area
    return dt_area, gt_area, inter_area


def calc_IoU(dt_xz, gt_xz, dt_height, gt_height):
    """
    :param dt_xz: Prediction boundaries can also be corners, format: [[x1, z1], [x2, z2], ...]
    :param gt_xz: Ground truth boundaries can also be corners, format: [[x1, z1], [x2, z2], ...]
    :param dt_height:
    :param gt_height:
    :return:
    """
    dt_area, gt_area, inter_area = calc_inter_area(dt_xz, gt_xz)
    iou_2d = inter_area / (gt_area + dt_area - inter_area)

    dt_volume = dt_area * dt_height
    gt_volume = gt_area * gt_height
    inter_volume = inter_area * min(dt_height, gt_height)
    iou_3d = inter_volume / (dt_volume + gt_volume - inter_volume)

    return iou_2d, iou_3d


def calc_Iou_height(dt_height, gt_height):
    return min(dt_height, gt_height) / max(dt_height, gt_height)



def calc_accuracy(pred_depth, pred_ratio, depth, ratio, h=512):
    visb_iou_2ds = []
    visb_iou_3ds = []
    # full_iou_2ds = []
    # full_iou_3ds = []
    iou_heights = []
    # rmse_s = []
    # delta_1_s = []

    visb_iou_floodplans = []


    for i in range(len(depth)):
        dt_xyz = depth2xyz(np.abs(pred_depth[i]))
        visb_gt_xyz = depth2xyz(np.abs(depth[i]))
        #corners = corner[i]
        #full_gt_corners = corners[corners[..., 0] + corners[..., 1] != 0]  # Take effective corners
        #full_gt_xyz = uv2xyz(full_gt_corners)

        dt_xz = dt_xyz[..., ::2]
        visb_gt_xz = visb_gt_xyz[..., ::2]
        #full_gt_xz = full_gt_xyz[..., ::2]

        gt_ratio = ratio[i][0]
        dt_ratio = pred_ratio[i][0]

        gt_boundaries = corners2boundaries(ratio[i], corners_xyz=depth2xyz(depth[i]), step=None, visible=False)
        dt_boundaries = corners2boundaries(pred_ratio[i], corners_xyz=dt_xyz, step=None, length=None, visible=False)
        # gt_layout_depth = layout2depth(gt_boundaries, show=False)
        # dt_layout_depth = layout2depth(dt_boundaries, show=False)

        visb_iou_2d, visb_iou_3d = calc_IoU(dt_xz, visb_gt_xz, dt_height=1 + dt_ratio, gt_height=1 + gt_ratio)
        #full_iou_2d, full_iou_3d = calc_IoU(dt_xz, full_gt_xz, dt_height=1 + dt_ratio, gt_height=1 + gt_ratio)
        iou_height = calc_Iou_height(dt_height=1 + dt_ratio, gt_height=1 + gt_ratio)

        # rmse = ((gt_layout_depth - dt_layout_depth) ** 2).mean() ** 0.5
        # threshold = np.maximum(gt_layout_depth / dt_layout_depth, dt_layout_depth / gt_layout_depth)
        # delta_1 = (threshold < 1.25).mean()

        visb_iou_2ds.append(visb_iou_2d)
        visb_iou_3ds.append(visb_iou_3d)
        #full_iou_2ds.append(full_iou_2d)
        #full_iou_3ds.append(full_iou_3d)
        iou_heights.append(iou_height)
        # rmse_s.append(rmse)
        # delta_1_s.append(delta_1)

    visb_iou_2d = np.array(visb_iou_2ds).mean()
    visb_iou_3d = np.array(visb_iou_3ds).mean()
    #full_iou_2d = np.array(full_iou_2ds).mean()
    #full_iou_3d = np.array(full_iou_3ds).mean()
    iou_height = np.array(iou_heights).mean()
    # rmse_s = np.array(rmse_s).mean()
    # delta_1_s = np.array(delta_1_s).mean()

    # return [visb_iou_2d, visb_iou_3d, visb_iou_floodplans], [full_iou_2d, full_iou_3d, full_iou_floodplans], iou_height, full_iou_2ds, rmse_s, delta_1_s
    return [visb_iou_2d, visb_iou_3d, visb_iou_floodplans], iou_height
