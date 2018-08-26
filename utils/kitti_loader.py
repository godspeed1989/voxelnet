#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import cv2
import numpy as np
import os
import sys
import glob
import math
import multiprocessing


from config import cfg
from utils.data_aug import aug_data
from utils.preprocess import process_pointcloud

class Processor:
    def __init__(self, data_tag, f_rgb, f_lidar, f_label, f_voxel, data_dir, aug, is_testset):
        self.data_tag=data_tag
        self.f_rgb = f_rgb
        self.f_lidar = f_lidar
        self.f_label = f_label
        self.f_voxel = f_voxel
        self.data_dir = data_dir
        self.aug = aug
        self.is_testset = is_testset

    def __call__(self,load_index):
        if self.aug:
            ret = aug_data(self.data_tag[load_index], self.data_dir)
        else:
            rgb = cv2.resize(cv2.imread(self.f_rgb[load_index]), (cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT))
            raw_lidar = np.fromfile(self.f_lidar[load_index], dtype=np.float32).reshape((-1, 4))
            if not self.is_testset:
                labels = [line for line in open(self.f_label[load_index], 'r').readlines()]
            else:
                labels = ['']
            tag = self.data_tag[load_index]
            if self.f_voxel is None:
                voxel = process_pointcloud(tag, raw_lidar)
            else:
                voxel_files = np.load(self.f_voxel[load_index])
                voxel = {}
                voxel['feature_buffer'] = voxel_files['feature_buffer']
                voxel['coordinate_buffer'] = voxel_files['coordinate_buffer']
                voxel['number_buffer'] = voxel_files['number_buffer']
                voxel['mask_buffer'] = voxel_files['mask_buffer']
            ret = [tag, rgb, raw_lidar, voxel, labels]
        return ret

# global pool
TRAIN_POOL = multiprocessing.Pool(4)
VAL_POOL = multiprocessing.Pool(2)

def iterate_data(data_dir, has_voxel=False, shuffle=False, aug=False,
                 is_testset=False, batch_size=1, multi_gpu_sum=1):
    f_rgb = glob.glob(os.path.join(data_dir, 'image_2', '*.png'))
    f_lidar = glob.glob(os.path.join(data_dir, 'velodyne', '*.bin'))
    f_label = glob.glob(os.path.join(data_dir, 'label_2', '*.txt'))
    if has_voxel:
        f_voxel = glob.glob(os.path.join(data_dir, 'voxel', '*.npz'))
        f_voxel.sort()
    else:
        f_voxel = None
    f_rgb.sort()
    f_lidar.sort()
    f_label.sort()
    data_tag = [name.split('/')[-1].split('.')[-2] for name in f_rgb]

    assert len(data_tag) != 0, "dataset folder is not correct"
    assert len(data_tag) == len(f_rgb) == len(f_lidar) == len(f_label), "dataset folder is not correct"
    if has_voxel: assert len(f_voxel) == len(f_label), "dataset folder is not correct"

    nums = len(f_rgb)
    indices = list(range(nums))
    if shuffle:
        np.random.shuffle(indices)

    num_batches = int(math.floor( nums / float(batch_size) ))

    proc=Processor(data_tag, f_rgb, f_lidar, f_label, f_voxel, data_dir, aug, is_testset)

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        excerpt = indices[start_idx:start_idx + batch_size]

        rets=TRAIN_POOL.map(proc,excerpt)

        tag = [ ret[0] for ret in rets ]
        rgb = [ ret[1] for ret in rets ]
        raw_lidar = [ ret[2] for ret in rets ]
        voxel = [ ret[3] for ret in rets ]
        labels = [ ret[4] for ret in rets ]

        # only for voxel -> [gpu, k_single_batch, ...]
        vox_feature, vox_number, vox_coordinate, vox_mask, vox_cnt = [], [], [], [], []
        single_batch_size = int(batch_size / multi_gpu_sum)
        for idx in range(multi_gpu_sum):
            _, per_vox_feature, per_vox_number, per_vox_coordinate, per_vox_mask, total_vox_cnt = \
                build_input(voxel[idx * single_batch_size:(idx + 1) * single_batch_size])
            vox_feature.append(per_vox_feature)
            vox_number.append(per_vox_number)
            vox_coordinate.append(per_vox_coordinate)
            vox_mask.append(per_vox_mask)
            vox_cnt.append(total_vox_cnt)

        ret = (
               np.array(tag),
               np.array(labels),
               np.array(vox_feature),
               np.array(vox_number),
               np.array(vox_coordinate),
               np.array(vox_mask),
               np.array(vox_cnt),
               np.array(rgb),
               np.array(raw_lidar)
               )

        yield ret

# Random sample single batch data
def sample_test_data(data_dir, batch_size=1, has_voxel=False, multi_gpu_sum=1):
    f_rgb = glob.glob(os.path.join(data_dir, 'image_2', '*.png'))
    f_lidar = glob.glob(os.path.join(data_dir, 'velodyne', '*.bin'))
    f_label = glob.glob(os.path.join(data_dir, 'label_2', '*.txt'))
    if has_voxel:
        f_voxel = glob.glob(os.path.join(data_dir, 'voxel', '*.npz'))
        f_voxel.sort()
    else:
        f_voxel = None
    f_rgb.sort()
    f_lidar.sort()
    f_label.sort()
    data_tag = [name.split('/')[-1].split('.')[-2] for name in f_rgb]

    assert len(data_tag) != 0, "dataset folder is not correct"
    assert len(data_tag) == len(f_rgb) == len(f_lidar) == len(f_label), "dataset folder is not correct"
    if has_voxel: assert len(f_voxel) == len(f_label), "dataset folder is not correct"

    nums = len(f_rgb)

    indices = list(range(nums))
    np.random.shuffle(indices)
    excerpt = indices[0:batch_size]

    proc_val=Processor(data_tag, f_rgb, f_lidar, f_label, None, data_dir, False, False)

    rets=VAL_POOL.map(proc_val,excerpt)

    tag = [ ret[0] for ret in rets ]
    rgb = [ ret[1] for ret in rets ]
    raw_lidar = [ ret[2] for ret in rets ]
    voxel = [ ret[3] for ret in rets ]
    labels = [ ret[4] for ret in rets ]

    # only for voxel -> [gpu, k_single_batch, ...]
    vox_feature, vox_number, vox_coordinate, vox_mask, vox_cnt = [], [], [], [], []
    single_batch_size = int(batch_size / multi_gpu_sum)
    for idx in range(multi_gpu_sum):
        _, per_vox_feature, per_vox_number, per_vox_coordinate, per_vox_mask, total_vox_cnt = \
            build_input(voxel[idx * single_batch_size:(idx + 1) * single_batch_size])
        vox_feature.append(per_vox_feature)
        vox_number.append(per_vox_number)
        vox_coordinate.append(per_vox_coordinate)
        vox_mask.append(per_vox_mask)
        vox_cnt.append(total_vox_cnt)

    ret = (
           np.array(tag),
           np.array(labels),
           np.array(vox_feature),
           np.array(vox_number),
           np.array(vox_coordinate),
           np.array(vox_mask),
           np.array(vox_cnt),
           np.array(rgb),
           np.array(raw_lidar)
           )

    return ret


def build_input(voxel_dict_list):
    batch_size = len(voxel_dict_list)

    feature_list = []
    number_list = []
    coordinate_list = []
    mask_list = []
    voxelcnt_list = []
    # combine batches to one
    for i, voxel_dict in zip(range(batch_size), voxel_dict_list):
        feature_list.append(voxel_dict['feature_buffer'])
        number_list.append(voxel_dict['number_buffer'])
        # (K, 3) -> (K, 4) pad batch number at start
        coordinate = voxel_dict['coordinate_buffer']
        coordinate = np.pad(coordinate, ((0, 0), (1, 0)), mode='constant', constant_values=i)
        coordinate_list.append(coordinate)
        mask_list.append(voxel_dict['mask_buffer'])
        voxelcnt_list.append([voxel_dict['feature_buffer'].shape[0]])

    feature = np.concatenate(feature_list)
    number = np.concatenate(number_list)
    coordinate = np.concatenate(coordinate_list)
    mask = np.concatenate(mask_list)
    voxelcnt = np.concatenate(voxelcnt_list)
    return batch_size, feature, number, coordinate, mask, voxelcnt


if __name__ == '__main__':
    pass
