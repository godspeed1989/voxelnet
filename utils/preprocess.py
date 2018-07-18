#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name : preprocess.py
# Purpose :
# Creation Date : 10-12-2017
# Last Modified : Thu 18 Jan 2018 05:34:42 PM CST
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]

import os
import multiprocessing
import numpy as np

from config import cfg

sensor_height_ = 2.5
num_iter_ = 15
th_dist_ = 0.1
stable_delta_ = 10

def extract_initial_seeds_(pc_velo_orig):
    pc_velo = pc_velo_orig[:, :3].copy()
    # Error point removal
    # As there are some error mirror reflection under the ground
    # here regardless point under 2* sensor_height
    pc_velo = pc_velo[pc_velo[:, 2] > -1.5*sensor_height_, :]
    # Sort along Z-axis
    pc_velo[pc_velo[:, 2].argsort()]
    # LPR is the mean of low point representative
    # lowest 1% points
    sum_cnt = (int)(pc_velo.shape[0] * 0.01)
    lpr_height = np.mean(pc_velo[:sum_cnt, 2])
    g_seeds_pc = pc_velo[pc_velo[:, 2] < lpr_height]
    return g_seeds_pc

def estimate_plane_(g_ground_pc):
    g_ground_pc = g_ground_pc[:, 0:3]
    # Create covarian matrix.
    # 1. calculate (x,y,z) mean (3,1)
    mean_xyz = np.expand_dims(np.mean(g_ground_pc, axis=0), axis=-1)
    # 2. calculate covariance
    # cov(x,x), cov(y,y), cov(z,z)
    # cov(x,y), cov(x,z), cov(y,z)
    cov = np.cov(g_ground_pc.T)
    # Singular Value Decomposition: SVD
    u, s, vh = np.linalg.svd(cov)
    # use the least singular vector as normal (3,1)
    normal_ = np.expand_dims(u[:,2], axis=-1)
    # according to normal.T*[x,y,z] = -d
    d_ = -np.dot(normal_.T, mean_xyz)[0,0]
    # set distance threhold to `th_dist - d`
    th_dist_d_ = - d_ + th_dist_
    return th_dist_d_, normal_

def filter_ground(pc_velo):
    # Extract init ground seeds.
    g_seeds_pc = extract_initial_seeds_(pc_velo)
    g_ground_pc = g_seeds_pc
    # Ground plane fitter mainloop
    for i in range(num_iter_):
        cnt_prev = g_ground_pc.shape[0]
        th_dist_d_, normal_ = estimate_plane_(g_ground_pc)
        # ground plane model (n,3)*(3,1)=(n,1)
        result = np.dot(pc_velo[:, 0:3], normal_)
        result = np.squeeze(result)
        # threshold filter
        g_ground_pc = pc_velo[result <= th_dist_d_, :]
        g_not_ground_pc = pc_velo[result > th_dist_d_, :]
        cnt_delta = abs(g_ground_pc.shape[0] - cnt_prev)
        if cnt_delta < stable_delta_:
            break
    return g_ground_pc, g_not_ground_pc

def process_pointcloud(tag, lidar, cls=cfg.DETECT_OBJ):
    # Input:
    #   (N, 4)
    # Output:
    #   voxel_dict
    if cls == 'Car':
        scene_size = np.array([4, 80, 70.4], dtype=np.float32)
        voxel_size = np.array([0.4, 0.2, 0.2], dtype=np.float32)
        grid_size = np.array([10, 400, 352], dtype=np.int64)
        lidar_coord = np.array([0, 40, 3], dtype=np.float32)
        max_point_number = 35
    else:
        scene_size = np.array([4, 40, 48], dtype=np.float32)
        voxel_size = np.array([0.4, 0.2, 0.2], dtype=np.float32)
        grid_size = np.array([10, 200, 240], dtype=np.int64)
        lidar_coord = np.array([0, 20, 3], dtype=np.float32)
        max_point_number = 45
    np.random.shuffle(lidar)

    if cfg.REMOVE_GROUND:
        _, point_cloud = filter_ground(lidar)
    assert point_cloud.shape[0], 'ERROR size {}'.format(tag)

    shifted_coord = point_cloud[:, :3] + lidar_coord
    # reverse the point cloud coordinate (X, Y, Z) -> (Z, Y, X)
    voxel_index = np.floor(
        shifted_coord[:, ::-1] / voxel_size).astype(np.int)

    bound_x = np.logical_and(
        voxel_index[:, 2] >= 0, voxel_index[:, 2] < grid_size[2])
    bound_y = np.logical_and(
        voxel_index[:, 1] >= 0, voxel_index[:, 1] < grid_size[1])
    bound_z = np.logical_and(
        voxel_index[:, 0] >= 0, voxel_index[:, 0] < grid_size[0])

    bound_box = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)

    point_cloud = point_cloud[bound_box]
    voxel_index = voxel_index[bound_box]

    # [K, 3] coordinate buffer as described in the paper
    coordinate_buffer = np.unique(voxel_index, axis=0)

    K = len(coordinate_buffer)
    T = max_point_number

    # [K, 1] store number of points in each voxel grid
    number_buffer = np.zeros(shape=(K), dtype=np.int64)

    # [K, T, 7] feature buffer as described in the paper
    feature_buffer = np.zeros(shape=(K, T, 7), dtype=np.float32)

    # build a reverse index for coordinate buffer
    index_buffer = {}
    for i in range(K):
        index_buffer[tuple(coordinate_buffer[i])] = i

    for voxel, point in zip(voxel_index, point_cloud):
        index = index_buffer[tuple(voxel)]
        number = number_buffer[index]
        if number < T:
            feature_buffer[index, number, :4] = point
            number_buffer[index] += 1

    feature_buffer[:, :, -3:] = feature_buffer[:, :, :3] - \
        feature_buffer[:, :, :3].sum(axis=1, keepdims=True)/number_buffer.reshape(K, 1, 1)

    voxel_dict = {'feature_buffer': feature_buffer,
                  'coordinate_buffer': coordinate_buffer,
                  'number_buffer': number_buffer}
    return voxel_dict

