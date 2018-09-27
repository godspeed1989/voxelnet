#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name : data_aug.py
# Purpose :
# Creation Date : 21-12-2017
# Last Modified : Fri 19 Jan 2018 01:06:35 PM CST
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]

import numpy as np
import cv2
import os
import multiprocessing as mp
import argparse
import glob

from utils.utils import *
from config import cfg
from utils.preprocess import process_pointcloud


def aug_data(tag, object_dir, aug_pc=True, use_newtag=False):
    np.random.seed()
    rgb = cv2.imread(os.path.join(object_dir, 'image_2', tag + '.png'))
    assert rgb is not None, print('ERROR rgb {} {}'.format(object_dir, tag))
    rgb = cv2.resize(rgb, (cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT))
    #
    lidar_path = os.path.join(object_dir, 'velodyne', tag + '.bin')
    lidar = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
    assert lidar.shape[0], print('ERROR lidar {} {}'.format(object_dir, tag))
    #
    label_path = os.path.join(object_dir, 'label_2', tag + '.txt')
    label = np.array([line for line in open(label_path, 'r').readlines()])  # (N')
    classes = np.array([line.split()[0] for line in label])  # (N')
    # (N', 7) x, y, z, h, w, l, r
    gt_box3d = label_to_gt_box3d([tag], np.array(label)[np.newaxis, :], cls='', coordinate='camera')[0]
    P, Tr, R = load_calib( os.path.join( cfg.CALIB_DIR, tag + '.txt' ) )
    if aug_pc:
        choice = np.random.randint(0, 10)
        if choice < 4:
            # global rotation
            angle = np.random.uniform(-np.pi / 30, np.pi / 30)
            lidar[:, 0:3] = point_transform(lidar[:, 0:3], 0, 0, 0, rz=angle)
            lidar_center_gt_box3d = camera_to_lidar_box(gt_box3d, T_VELO_2_CAM=Tr, R_RECT_0=R)
            lidar_center_gt_box3d = box_transform(lidar_center_gt_box3d, 0, 0, 0, r=angle, coordinate='lidar')
            gt_box3d = lidar_to_camera_box(lidar_center_gt_box3d, T_VELO_2_CAM=Tr, R_RECT_0=R)
            newtag = 'aug_{}_1_{:.4f}'.format(tag, angle).replace('.', '_')
        elif choice < 7:
            # global translation
            tx = np.random.uniform(-0.1, -0.1)
            ty = np.random.uniform(-0.1, -0.1)
            tz = np.random.uniform(-0.15, -0.15)
            lidar[:, 0:3] = point_transform(lidar[:, 0:3], tx, ty, tz)
            lidar_center_gt_box3d = camera_to_lidar_box(gt_box3d, T_VELO_2_CAM=Tr, R_RECT_0=R)
            lidar_center_gt_box3d = box_transform(lidar_center_gt_box3d, tx, ty, tz, coordinate='lidar')
            gt_box3d = lidar_to_camera_box(lidar_center_gt_box3d, T_VELO_2_CAM=Tr, R_RECT_0=R)
            newtag = 'aug_{}_2_trans'.format(tag).replace('.', '_')
        else:
            # global scaling
            factor = np.random.uniform(0.95, 1.05)
            lidar[:, 0:3] = lidar[:, 0:3] * factor
            lidar_center_gt_box3d = camera_to_lidar_box(gt_box3d, T_VELO_2_CAM=Tr, R_RECT_0=R)
            lidar_center_gt_box3d[:, 0:6] = lidar_center_gt_box3d[:, 0:6] * factor
            gt_box3d = lidar_to_camera_box(lidar_center_gt_box3d, T_VELO_2_CAM=Tr, R_RECT_0=R)
            newtag = 'aug_{}_3_{:.4f}'.format(tag, factor).replace('.', '_')
    else:
        newtag = tag

    label = box3d_to_label(gt_box3d[np.newaxis, ...], classes[np.newaxis, ...], coordinate='camera',
                           P2=P, T_VELO_2_CAM=Tr, R_RECT_0=R)[0]  # (N')
    voxel_dict = process_pointcloud(tag, lidar)
    if use_newtag:
        return newtag, rgb, lidar, voxel_dict, label
    else:
        return tag, rgb, lidar, voxel_dict, label