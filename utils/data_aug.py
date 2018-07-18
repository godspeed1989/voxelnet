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
from utils.preprocess import process_pointcloud


def aug_data(tag, object_dir):
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
    cls = np.array([line.split()[0] for line in label])  # (N')
    # (N', 7) x, y, z, h, w, l, r
    gt_box3d = label_to_gt_box3d(np.array(label)[np.newaxis, :], cls='', coordinate='camera')[0]

    choice = np.random.randint(0, 10)
    if choice < 5:
        # global rotation
        angle = np.random.uniform(-np.pi / 4, np.pi / 4)
        lidar[:, 0:3] = point_transform(lidar[:, 0:3], 0, 0, 0, rz=angle)
        lidar_center_gt_box3d = camera_to_lidar_box(gt_box3d)
        lidar_center_gt_box3d = box_transform(lidar_center_gt_box3d, 0, 0, 0, r=angle, coordinate='lidar')
        gt_box3d = lidar_to_camera_box(lidar_center_gt_box3d)
        newtag = 'aug_{}_2_{:.4f}'.format(tag, angle).replace('.', '_')
    else:
        # global scaling
        factor = np.random.uniform(0.95, 1.05)
        lidar[:, 0:3] = lidar[:, 0:3] * factor
        lidar_center_gt_box3d = camera_to_lidar_box(gt_box3d)
        lidar_center_gt_box3d[:, 0:6] = lidar_center_gt_box3d[:, 0:6] * factor
        gt_box3d = lidar_to_camera_box(lidar_center_gt_box3d)
        newtag = 'aug_{}_3_{:.4f}'.format(tag, factor).replace('.', '_')

    label = box3d_to_label(gt_box3d[np.newaxis, ...], cls[np.newaxis, ...], coordinate='camera')[0]  # (N')
    voxel_dict = process_pointcloud(tag, lidar)
    return newtag, rgb, lidar, voxel_dict, label


