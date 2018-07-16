#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
import time
from model.PntSIFT.pointSIFT_util import pointSIFT_module

from config import cfg


class VFELayer(object):

    def __init__(self, out_channels, name):
        super(VFELayer, self).__init__()
        self.out_channels = out_channels
        self.name = name

    def apply(self, inputs, mask, training):
        radius = cfg.VOXEL_X_SIZE / 2.
        _, out_feature, _ = pointSIFT_module(inputs, None, radius, self.out_channels, training, scope=self.name)

        mask = tf.tile(mask, [1, 1, self.out_channels])
        out_feature = tf.multiply(out_feature, tf.cast(mask, tf.float32))
        return out_feature

class FeatureNetSIFT(object):

    def __init__(self, training, batch_size, name=''):
        super(FeatureNetSIFT, self).__init__()
        self.training = training

        # scalar
        self.batch_size = batch_size
        # [ΣK, 35/45, 7]
        self.feature_pl = tf.placeholder(
            tf.float32, [None, cfg.VOXEL_POINT_COUNT, 7], name='feature')
        # [ΣK]
        self.number_pl = tf.placeholder(tf.int64, [None], name='number')
        # [ΣK, 4], each row stores (batch, d, h, w)
        self.coordinate_pl = tf.placeholder(
            tf.int64, [None, 4], name='coordinate')

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            self.vfe1 = VFELayer(32, 'VFE-SIFT-1')
            self.vfe2 = VFELayer(128, 'VFE-SIFT-2')

        self.feature = self.feature_pl[..., :3]
        # boolean mask [K, T, 2 * units]
        mask = tf.not_equal(tf.reduce_max(
            self.feature, axis=2, keepdims=True), 0)
        x = self.vfe1.apply(self.feature, mask, self.training)
        x = self.vfe2.apply(x, mask, self.training)

        # [ΣK, 128]
        voxelwise = tf.reduce_max(x, axis=1)

        # car: [N * 10 * 400 * 352 * 128]
        # pedestrian/cyclist: [N * 10 * 200 * 240 * 128]
        self.outputs = tf.scatter_nd(
            self.coordinate_pl, voxelwise, [self.batch_size, 10, cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH, 128])
