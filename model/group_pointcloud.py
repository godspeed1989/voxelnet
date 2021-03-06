#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
import time

from config import cfg


class VFELayer(object):

    def __init__(self, out_channels, name):
        super(VFELayer, self).__init__()
        self.units = int(out_channels / 2)
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            self.dense = tf.layers.Dense(
                self.units, tf.nn.relu, name='dense', _reuse=tf.AUTO_REUSE, _scope=scope)
            self.batch_norm = tf.layers.BatchNormalization(
                name='batch_norm', fused=True, _reuse=tf.AUTO_REUSE, _scope=scope)

    def apply(self, inputs, mask, training):
        # [K, T, F] tensordot [F, units] -> [K, T, units]
        pointwise = self.batch_norm.apply(self.dense.apply(inputs), training)

        pointwise = tf.nn.relu(pointwise)
        # [K, 1, units]
        aggregated = tf.reduce_max(pointwise, axis=1, keepdims=True)

        # [K, T, units]
        repeated = tf.tile(aggregated, [1, cfg.VOXEL_POINT_COUNT, 1])

        # [K, T, 2 * units]
        concatenated = tf.concat([pointwise, repeated], axis=2)

        mask = tf.tile(mask, [1, 1, 2 * self.units])

        concatenated = tf.multiply(concatenated, tf.cast(mask, tf.float32))

        return concatenated


class FeatureNet(object):

    def __init__(self, training, batch_size, name=''):
        super(FeatureNet, self).__init__()
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

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            self.vfe1 = VFELayer(32, 'VFE-1')
            self.vfe2 = VFELayer(128, 'VFE-2')

        # boolean mask [K, T, 2 * units]
        #mask = tf.not_equal(tf.reduce_max(
        #    self.feature_pl, axis=2, keepdims=True), 0)
        x = self.vfe1.apply(self.feature_pl, self.mask_pl, self.training)
        x = self.vfe2.apply(x, self.mask_pl, self.training)

        # [K, 128]
        voxelwise = tf.reduce_max(x, axis=1)

        self.outputs = tf.scatter_nd(
            self.coordinate_pl, voxelwise, [self.batch_size, cfg.GRID_Z_SIZE, cfg.GRID_Y_SIZE, cfg.GRID_X_SIZE, 128])
