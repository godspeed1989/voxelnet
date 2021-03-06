#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
import time

if __name__ != '__main__':
    from config import cfg

''' Compute pairwise distance of a point cloud
input
    A shape is (N, P_A, C), B shape is (N, P_B, C)
return
    D shape is (N, P_A, P_B)
'''
def batch_distance_matrix_general(A, B=None):
    with tf.variable_scope('batch_distance_matrix_general'):
        if B is None:
            B = A
        r_A = tf.reduce_sum(A * A, axis=2, keepdims=True)
        r_B = tf.reduce_sum(B * B, axis=2, keepdims=True)
        m = tf.matmul(A, tf.transpose(B, perm=(0, 2, 1)))
        D = r_A - 2 * m + tf.transpose(r_B, perm=(0, 2, 1))
    return D

""" Get KNN based on the pairwise distance
Args:
    pairwise distance: (batch_size, num_points, num_points)
    k: int
Returns:
    nearest neighbors: (batch_size, num_points, k)
"""
def knn(adj_matrix, k):
    neg_adj = -adj_matrix
    _, nn_idx = tf.nn.top_k(neg_adj, k=k, sorted=True)
    return tf.cast(nn_idx, tf.int64)

""" Construct nearest neighbor for each point
Args:
    pts: (B, N, F)
    nn_idx: (B, N, K) 每个点临近k个点的索引
Returns:
    (B, N, K, F)
"""
def get_nn(pts, batch_size, nn_idx):
    num_points = pts.get_shape()[1].value
    num_dims = pts.get_shape()[2].value
    pts_flat = tf.reshape(pts, [-1, num_dims]) # flatten

    # [0*P, 1*P,..., (B-1)*P]
    batch_idx_ = tf.range(batch_size, dtype=tf.int64) * num_points
    # [[[0]], [[P]], ..., [[(B-1)*P]]] = (B, 1, 1)
    batch_idx_ = tf.expand_dims(batch_idx_, axis=-1)
    batch_idx_ = tf.expand_dims(batch_idx_, axis=-1)
    # 因为flatten了point_cloud，所以需要加上batch_idx
    # indices = (B, 1, 1) + (B, N, K) = (B, N, K)
    neighbors_indices = nn_idx + batch_idx_
    # (B, N, K, F)
    pts_neighbors = tf.gather(pts_flat, neighbors_indices)
    return pts_neighbors

def GetNNFeature(pts, K, batch_size, tag):
    with tf.variable_scope(tag):
        # get nearest neighbors index (B, N, K)
        adj_matrix = batch_distance_matrix_general(pts)
        nn_idx = knn(adj_matrix, k=K)
        # extract neighbor (B, N, C) -> (B, N, K, C)
        nn_feature = get_nn(pts, batch_size, nn_idx=nn_idx)
        return nn_feature

def pointcnv(M, pc_feature, Cout, scope, training, activation=True):
    """
    Input: (B,N,C)
    Return: (B,N,Cout)
    """
    with tf.variable_scope(scope):
        if M == 1:
            pc_feature = tf.layers.conv1d(pc_feature, Cout, kernel_size=1, strides=1, use_bias=False,
                            padding="valid", reuse=tf.AUTO_REUSE, name='conv1d')
        elif M == 2:
            pc_feature = tf.layers.conv2d(pc_feature, Cout, kernel_size=1, strides=1, use_bias=False,
                            padding="valid", reuse=tf.AUTO_REUSE, name='conv2d')
        pc_feature = tf.layers.batch_normalization(pc_feature, axis=-1,
                        fused=True, training=training, reuse=tf.AUTO_REUSE, name='bn')
        if activation:
            return tf.nn.relu(pc_feature)
        else:
            return pc_feature

class VFELayer(object):
    def __init__(self, batch_size, out_channels, name):
        super(VFELayer, self).__init__()
        self.units = out_channels
        self.batch_size = batch_size

    def apply(self, inputs, mask, training):
        if False:
            nn = 4
            nn_feature = GetNNFeature(inputs, nn, self.batch_size, 'nn_feature')
            nn_feature = pointcnv(2, nn_feature, 32, 'vfe_nn_conv1', training)
            nn_feature = pointcnv(2, nn_feature, 64, 'vfe_nn_conv2', training, activation=False)
            nn_feature = tf.reduce_max(nn_feature, axis=2, keepdims=False)
            nn_feature = tf.multiply(nn_feature, tf.cast(mask64, tf.float32))
        else:
            nn_feature = inputs

        feature = pointcnv(1, inputs, 32, 'vfe_conv1', training)
        feature = pointcnv(1, feature, 64, 'vfe_conv2', training)
        feature = pointcnv(1, feature, self.units, 'vfe_out', training, activation=False)

        # [K, T, units] -> [K, units]
        aggregated = tf.reduce_max(feature, axis=1, keepdims=False)

        return aggregated

class FeatureNet_PntNet1(object):

    def __init__(self, training, batch_size, name=''):
        super(FeatureNet_PntNet1, self).__init__()
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

        total_voxels = tf.reduce_sum(self.voxcnt_pl)
        Cout = 128
        self.vfe = VFELayer(total_voxels, Cout, 'VFE')
        voxelwise = self.vfe.apply(self.feature_pl, self.mask_pl, self.training)

        #max_intensity = tf.reduce_max(self.feature_pl[:,:,3], axis=-1, keepdims=True)
        #number_vox = tf.expand_dims(tf.cast(self.number_pl, tf.float32), axis=-1) / cfg.VOXEL_POINT_COUNT
        #voxelwise = tf.concat((voxelwise, max_intensity, number_vox), axis=-1)

        self.outputs = tf.scatter_nd(
            self.coordinate_pl, voxelwise, [self.batch_size, cfg.GRID_Z_SIZE, cfg.GRID_Y_SIZE, cfg.GRID_X_SIZE, Cout])

if __name__ == '__main__':
    VOXEL_POINT_COUNT = 50
    training = tf.placeholder(tf.bool)
    # [K, T, F]
    feature_pl = tf.placeholder(tf.float32, [None, VOXEL_POINT_COUNT, 4], name='feature')
    # [K]
    number_pl = tf.placeholder(tf.int64, [None], name='number')
    # []
    voxcnt_pl = tf.placeholder(tf.int64, [None], name='total_voxel_cnt')
    # [K, T, 1]
    mask_pl = tf.placeholder(tf.bool, [None, VOXEL_POINT_COUNT, 1], name='mask')

    total_voxels = tf.reduce_sum(voxcnt_pl)
    Cout = 128
    vfe = VFELayer(total_voxels, Cout, 'VFE')
    voxelwise = vfe.apply(feature_pl[:,:,:3], mask_pl, training)

    max_intensity = tf.reduce_max(feature_pl[:,:,3], axis=-1, keepdims=True)
    number_vox = tf.expand_dims(tf.cast(number_pl, tf.float32), axis=-1) / VOXEL_POINT_COUNT
    voxelwise = tf.concat((voxelwise, max_intensity, number_vox), axis=-1)

    voxels_total = 32
    feature_in = np.random.rand(voxels_total, VOXEL_POINT_COUNT, 4)
    number_in = np.ones([voxels_total,], dtype=np.int64)
    voxcnt_in = np.array([12, 20], dtype=np.int64)
    mask_in = np.ones([voxels_total, VOXEL_POINT_COUNT, 1], dtype=np.bool)
    #
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    ret = sess.run(voxelwise, {feature_pl: feature_in,
                               number_pl: number_in,
                               voxcnt_pl: voxcnt_in,
                               mask_pl: mask_in,
                               training: False})
    print(ret.shape)