#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
import time

from config import cfg

def pointcnv(pc_feature, Cout, scope, training, activation=True):
    """
    Input: (B,N,C)
    Return: (B,N,Cout)
    """
    with tf.variable_scope(scope):
        pc_feature = tf.layers.conv1d(pc_feature, Cout, kernel_size=1, strides=1,
                        padding="valid", reuse=tf.AUTO_REUSE, name='conv1d')
        pc_feature = tf.layers.batch_normalization(pc_feature, axis=-1,
                        fused=True, training=training, reuse=tf.AUTO_REUSE, name='bn')
        if activation:
            return tf.nn.relu(pc_feature)
        else:
            return pc_feature

class VFELayer(object):
    def __init__(self, out_channels, name):
        super(VFELayer, self).__init__()
        self.units = out_channels

    def apply(self, inputs, mask, training):
        pc_feature = pointcnv(inputs, 32, 'vfe_conv1', training)
        pc_feature = pointcnv(pc_feature, 64, 'vfe_conv2', training)
        pc_feature = tf.concat([inputs, pc_feature], axis=-1)
        pc_feature = pointcnv(pc_feature, 128, 'vfe_conv3', training)
        pc_feature = pointcnv(pc_feature, self.units, 'vfe_conv4', training, activation=False)

        # [K, T, 1] -> [K, T, units]
        mask = tf.tile(mask, [1, 1, self.units])
        pc_feature = tf.multiply(pc_feature, tf.cast(mask, tf.float32))
        # [K, T, units] -> [K, units]
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

        Cout = 128
        self.vfe = VFELayer(Cout, 'VFE')
        voxelwise = self.vfe.apply(self.feature_pl, self.mask_pl, self.training)

        self.outputs = tf.scatter_nd(
            self.coordinate_pl, voxelwise, [self.batch_size, cfg.GRID_Z_SIZE, cfg.GRID_Y_SIZE, cfg.GRID_X_SIZE, Cout])
