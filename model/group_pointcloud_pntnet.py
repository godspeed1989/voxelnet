#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
import time
import utils.tf_util as tf_util

from config import cfg

class VFELayer(object):
    def __init__(self, out_channels, name):
        super(VFELayer, self).__init__()
        self.units = out_channels

    def apply(self, inputs, mask, training):
        # [K, T, F] -> [K, T, 1, F]
        pc_feature = tf.expand_dims(inputs, axis=-2)
        pc_feature = tf_util.conv2d(pc_feature, 32, [1,1],
                                    padding='VALID', stride=[1,1],
                                    bn=True, is_training=training,
                                    scope='vfe_conv1',
                                    activation_fn=tf.nn.relu)
        pc_feature = tf_util.conv2d(pc_feature, 64, [1,1],
                                    padding='VALID', stride=[1,1],
                                    bn=True, is_training=training,
                                    scope='vfe_conv2',
                                    activation_fn=tf.nn.relu)
        pc_feature = tf_util.conv2d(pc_feature, 128, [1,1],
                                    padding='VALID', stride=[1,1],
                                    bn=True, is_training=training,
                                    scope='vfe_conv3',
                                    activation_fn=tf.nn.relu)
        pc_feature = tf_util.conv2d(pc_feature, self.units, [1,1],
                                    padding='VALID', stride=[1,1],
                                    bn=True, is_training=training,
                                    scope='vfe_conv4',
                                    activation_fn=tf.nn.relu)
        # [K, T, 1, C] -> [K, T, C]
        pc_feature = tf.squeeze(pc_feature, axis=2)
        # [K, 1, units]
        mask = tf.tile(mask, [1, 1, self.units])
        pc_feature = tf.multiply(pc_feature, tf.cast(mask, tf.float32))
        aggregated = tf.reduce_max(pc_feature, axis=1, keepdims=False)

        return aggregated

class FeatureNet_PntNet(object):

    def __init__(self, training, batch_size, name=''):
        super(FeatureNet_PntNet, self).__init__()
        self.training = training

        # scalar
        self.batch_size = batch_size
        # [K, T, F]
        self.feature_pl = tf.placeholder(tf.float32, [None, cfg.VOXEL_POINT_COUNT, cfg.POINT_FEATURE_LEN], name='feature')
        # [K]
        self.number_pl = tf.placeholder(tf.int64, [None], name='number')
        # [K, T, 1]
        self.mask_pl = tf.placeholder(tf.bool, [None, cfg.VOXEL_POINT_COUNT, 1], name='mask')
        # [K, 4], each row stores (batch, d, h, w)
        self.coordinate_pl = tf.placeholder(tf.int64, [None, 4], name='coordinate')

        self.vfe = VFELayer(256, 'VFE')
        voxelwise = self.vfe.apply(self.feature_pl, self.mask_pl, self.training)

        self.outputs = tf.scatter_nd(
            self.coordinate_pl, voxelwise, [self.batch_size, cfg.GRID_Z_SIZE, cfg.GRID_Y_SIZE, cfg.GRID_X_SIZE, 256])
