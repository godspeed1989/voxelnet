#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
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
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(BASE_DIR))

from pntae.model_ae import ae_encoder

class FeatureNet_AE(object):

    def __init__(self, training, batch_size, trainable=True):
        super(FeatureNet_AE, self).__init__()
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

        codec = ae_encoder(self.feature_pl[:,:,:3], self.mask_pl, self.training, trainable=trainable)
        max_intensity = tf.reduce_max(self.feature_pl[:,:,3], axis=-1, keepdims=True)
        self.voxelwise = tf.concat([codec, max_intensity], axis=-1)

        Cout = self.voxelwise.get_shape()[-1].value
        self.outputs = tf.scatter_nd(
            self.coordinate_pl, self.voxelwise, [self.batch_size, cfg.GRID_Z_SIZE, cfg.GRID_Y_SIZE, cfg.GRID_X_SIZE, Cout])

if __name__ == '__main__':
    training = tf.placeholder(tf.bool)
    ae = FeatureNet_AE(training, 3)

    voxels_total = 32
    feature_in = np.random.rand(voxels_total, cfg.VOXEL_POINT_COUNT, cfg.POINT_FEATURE_LEN)
    number_in = np.ones([voxels_total,], dtype=np.int64)
    voxcnt_in = np.array([12, 20], dtype=np.int64)
    mask_in = np.ones([voxels_total, cfg.VOXEL_POINT_COUNT, 1], dtype=np.bool)
    #
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    ret = sess.run(ae.voxelwise, {ae.feature_pl: feature_in,
                                  ae.number_pl: number_in,
                                  ae.voxcnt_pl: voxcnt_in,
                                  ae.mask_pl: mask_in,
                                  ae.training: False})
    print(ret.shape)