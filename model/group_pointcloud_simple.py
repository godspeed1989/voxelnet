#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
import time

if __name__ != '__main__':
    from config import cfg
else:
    from easydict import EasyDict as edict
    cfg = edict()
    cfg.VOXEL_POINT_COUNT = 50
    cfg.POINT_FEATURE_LEN = 6
    cfg.GRID_Z_SIZE, cfg.GRID_Y_SIZE, cfg.GRID_X_SIZE = 1, 100, 200

class FeatureNet_Simple(object):

    def __init__(self, training, batch_size, name=''):
        super(FeatureNet_Simple, self).__init__()
        self.training = training

        # scalar
        self.batch_size = batch_size
        # [K, T, F]
        self.feature_pl = tf.placeholder(tf.float32, [None, cfg.VOXEL_POINT_COUNT, cfg.POINT_FEATURE_LEN], name='feature')
        # [K]
        self.number_pl = tf.placeholder(tf.int64, [None], name='number')
        # []
        self.voxcnt_pl = tf.placeholder(tf.int64, [None], name='total_voxel_cnt')
        # [K, T, 1]
        self.mask_pl = tf.placeholder(tf.bool, [None, cfg.VOXEL_POINT_COUNT, 1], name='mask')
        # [K, 4], each row stores (batch, d, h, w)
        self.coordinate_pl = tf.placeholder(tf.int64, [None, 4], name='coordinate')

        min_z = tf.reduce_min(self.feature_pl[:,:,2], axis=-1, keepdims=True)
        max_z = tf.reduce_max(self.feature_pl[:,:,2], axis=-1, keepdims=True)
        max_intensity = tf.reduce_max(self.feature_pl[:,:,3], axis=-1, keepdims=True)
        mean_intensity = tf.reduce_sum(self.feature_pl[:,:,3], axis=-1, keepdims=True)
        mean_intensity = mean_intensity / tf.reduce_sum(tf.cast(self.mask_pl, tf.float32), axis=1, keepdims=False)

        number_vox = tf.expand_dims(tf.cast(self.number_pl, tf.float32), axis=-1) / cfg.VOXEL_POINT_COUNT
        self.voxelwise = tf.concat((min_z, max_z, max_intensity, mean_intensity, number_vox), axis=-1)
        Cout = self.voxelwise.get_shape()[-1]

        self.outputs = tf.scatter_nd(
            self.coordinate_pl, self.voxelwise, [self.batch_size, cfg.GRID_Z_SIZE, cfg.GRID_Y_SIZE, cfg.GRID_X_SIZE, Cout])

if __name__ == '__main__':
    training = tf.placeholder(tf.bool)

    fns = FeatureNet_Simple(training, 2)

    voxels_total = 32
    feature_in = np.random.rand(voxels_total, cfg.VOXEL_POINT_COUNT, cfg.POINT_FEATURE_LEN)
    number_in = np.ones([voxels_total,], dtype=np.int64)
    voxcnt_in = np.array([12, 20], dtype=np.int64)
    mask_in = np.ones([voxels_total, cfg.VOXEL_POINT_COUNT, 1], dtype=np.bool)
    #
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    ret = sess.run(fns.voxelwise, {fns.feature_pl: feature_in,
                                   fns.number_pl: number_in,
                                   fns.voxcnt_pl: voxcnt_in,
                                   fns.mask_pl: mask_in,
                                   fns.training: False})
    print(ret.shape)