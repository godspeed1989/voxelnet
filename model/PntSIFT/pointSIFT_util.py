"""
wrappers for pointSIFT module
Author: Jiang Mingyang
Email: jmydurant@sjtu.edu.cn
"""
import os
import sys
import numpy as np
import tensorflow as tf
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from pointSIFT_op.pointSIFT_op import pointSIFT_select, pointSIFT_select_four
from grouping_op.tf_grouping import group_point, query_ball_point, knn_point
import tf_util

def pointSIFT_group(radius, xyz, points, use_xyz=True):
    idx = pointSIFT_select(xyz, radius)
    grouped_xyz = group_point(xyz, idx)  # (batch_size, npoint, 8, 3)
    grouped_xyz -= tf.tile(tf.expand_dims(xyz, 2), [1, 1, 8, 1])  # translation normalization
    if points is not None:
        grouped_points = group_point(points, idx)  # (batch_size, npoint, 8, channel)
        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1)  # (batch_size, npoint, 8/32, 3+channel)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return xyz, new_points, idx, grouped_xyz

def pointSIFT_group_with_idx(xyz, idx, points, use_xyz=True):
    grouped_xyz = group_point(xyz, idx)  # (batch_size, npoint, 8, 3)
    grouped_xyz -= tf.tile(tf.expand_dims(xyz, 2), [1, 1, 8, 1])  # translation normalization
    if points is not None:
        grouped_points = group_point(points, idx)  # (batch_size, npoint, 8/32, channel)
        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1)  # (batch_size, npoint, 8/32, 3+channel)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz
    return xyz, new_points, idx, grouped_xyz

def pointSIFT_module(xyz, points, radius, out_channel, is_training, scope='point_sift', bn_decay=None, bn=True, use_xyz=True, use_nchw=False):
    data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.variable_scope(scope) as sc:
        # Grouping
        new_xyz, new_points, idx, grouped_xyz = pointSIFT_group(radius, xyz, points, use_xyz)

        # Point Feature Embedding
        if use_nchw: new_points = tf.transpose(new_points, [0, 3, 1, 2])
        for i in range(3):
            new_points = tf_util.conv2d(new_points, out_channel, [1, 2],
                                        padding='VALID', stride=[1, 2],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d' % (i), bn_decay=bn_decay,
                                        data_format=data_format)
        # add fc
        new_points = tf_util.conv2d(new_points, out_channel, [1, 1],
                                    padding='VALID', stride=[1, 1],
                                    bn=bn, is_training=is_training,
                                    scope='conv_fc', bn_decay=bn_decay,
                                    data_format=data_format)
        if use_nchw: new_points = tf.transpose(new_points, [0, 2, 3, 1])

        new_points = tf.squeeze(new_points, [2])  # (batch_size, npoints, mlp2[-1])
        return new_xyz, new_points, idx

def pointSIFT_res_module(xyz, points, radius, out_channel, is_training, bn_decay, scope='point_sift', bn=True, use_xyz=True, same_dim=False, merge='add'):
    data_format = 'NHWC'
    with tf.variable_scope(scope) as sc:
        # conv1
        _, new_points, idx, _ = pointSIFT_group(radius, xyz, points, use_xyz=use_xyz)

        for i in range(3):
            new_points = tf_util.conv2d(new_points, out_channel, [1, 2],
                                        padding='VALID', stride=[1, 2],
                                        bn=bn, is_training=is_training,
                                        scope='c0_conv%d' % (i), bn_decay=bn_decay,
                                        data_format=data_format)
        new_points = tf.squeeze(new_points, [2])
        # conv2
        _, new_points, idx, _ = pointSIFT_group_with_idx(xyz, idx=idx, points=new_points, use_xyz=use_xyz)

        for i in range(3):
            if i == 2:
                act = None
            else:
                act = tf.nn.relu
            new_points = tf_util.conv2d(new_points, out_channel, [1, 2],
                                        padding='VALID', stride=[1, 2],
                                        bn=bn, is_training=is_training,
                                        scope='c1_conv%d' % (i), bn_decay=bn_decay,
                                        activation_fn=act,
                                        data_format=data_format)
        new_points = tf.squeeze(new_points, [2])
        # residual part..
        if points is not None:
            if same_dim is True:
                points = tf_util.conv1d(points, out_channel, 1, padding='VALID', bn=bn, is_training=is_training, scope='merge_channel_fc', bn_decay=bn_decay)
            if merge == 'add':
                new_points = new_points + points
            elif merge == 'concat':
                new_points = tf.concat([new_points, points], axis=-1)
            else:
                print("ways not found!!!")
        new_points = tf.nn.relu(new_points)
        return xyz, new_points, idx

if __name__ == '__main__':
    batch_size = 4
    num_point = 128
    output_dim = 256
    radius = 0.1
    # input coordinates
    xyz = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    # input features
    point_feature = None
    # setting phases
    is_training = tf.placeholder(dtype=tf.bool, shape=())
    # setting searching radius (0.1 as an example)
    radius = 0.1
    _, out_feature, _ = pointSIFT_module(xyz, point_feature, radius, output_dim, is_training)
    with tf.Session() as sess:
        points = np.random.rand(batch_size, num_point, 3)
        sess.run(tf.global_variables_initializer())
        result = sess.run(out_feature, feed_dict = {
            xyz: points, is_training: False
        })
        print(result.shape)
        print(result[0, 32, 100:128])
