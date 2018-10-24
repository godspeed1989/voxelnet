#!/usr/bin/env python
# -*- coding:UTF-8 -*-
import tensorflow as tf
import numpy as np

from config import cfg

small_addon_for_BCE = 1e-8


class MiddleAndRPN:
    def __init__(self, input_data, alpha=1.5, beta=1, sigma=3, training=True, name=''):
        # scale should be the output of feature learning network
        self.input_data = input_data
        self.training = training
        # groundtruth(target) - each anchor box, represent as △x, △y, △z, △h, △w, △l, rotation
        self.targets = tf.placeholder(
            tf.float32, [None, cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH, cfg.ANCHOR_TYPES * cfg.ANCHOR_LEN])
        # postive anchors equal to one and others equal to zero
        self.pos_equal_one = tf.placeholder(
            tf.float32, [None, cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH, cfg.ANCHOR_TYPES])
        self.pos_equal_one_sum = tf.placeholder(tf.float32, [None, 1, 1, 1])
        self.pos_equal_one_for_reg = tf.placeholder(
            tf.float32, [None, cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH, cfg.ANCHOR_TYPES * cfg.ANCHOR_LEN])
        # negative anchors equal to one and others equal to zero
        self.neg_equal_one = tf.placeholder(
            tf.float32, [None, cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH, cfg.ANCHOR_TYPES])
        self.neg_equal_one_sum = tf.placeholder(tf.float32, [None, 1, 1, 1])

        # ConvMD(M, Cin, Cout, k, (s), (p), input, training, ...)
        with tf.variable_scope('MiddleAndRPN_' + name):
            # convolutinal middle layers
            temp_conv = self.input_data
            batch_size = self.input_data.get_shape()[0]
            if cfg.VOXEL_Z_ONE is not True and cfg.FEATURE_NET_TYPE != 'FeatureNet_Simple':
                temp_conv = ConvMD(3, 128, 64, 3, (2, 1, 1),
                                (1, 1, 1), temp_conv, training=self.training, name='conv1')
                temp_conv = ConvMD(3, 64, 64, 3, (1, 1, 1),
                                (0, 1, 1), temp_conv, training=self.training, name='conv2')
                temp_conv = ConvMD(3, 64, 64, 3, (2, 1, 1),
                                (1, 1, 1), temp_conv, training=self.training, name='conv3')
                print(temp_conv.shape)
            # (N, D, H, W, C) -> (N, H, W, C, D)
            temp_conv = tf.transpose(temp_conv, perm=[0, 2, 3, 4, 1])
            assert temp_conv.get_shape()[1] == cfg.INPUT_HEIGHT
            assert temp_conv.get_shape()[2] == cfg.INPUT_WIDTH
            temp_conv = tf.reshape(temp_conv, [batch_size, cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH, -1])

            if cfg.RPN_TYPE == 'voxelnet':
                # rpn
                # block1:
                temp_conv = ConvMD(2, 128, 128, 3, (2, 2), (1, 1),
                                temp_conv, training=self.training, name='conv4')
                temp_conv = ConvMD(2, 128, 128, 3, (1, 1), (1, 1),
                                temp_conv, training=self.training, name='conv5')
                temp_conv = ConvMD(2, 128, 128, 3, (1, 1), (1, 1),
                                temp_conv, training=self.training, name='conv6')
                temp_conv = ConvMD(2, 128, 128, 3, (1, 1), (1, 1),
                                temp_conv, training=self.training, name='conv7')
                deconv1 = Deconv2D(128, 256, 3, (1, 1), (0, 0),
                                temp_conv, training=self.training, name='deconv1')

                # block2:
                temp_conv = ConvMD(2, 128, 128, 3, (2, 2), (1, 1),
                                temp_conv, training=self.training, name='conv8')
                temp_conv = ConvMD(2, 128, 128, 3, (1, 1), (1, 1),
                                temp_conv, training=self.training, name='conv9')
                temp_conv = ConvMD(2, 128, 128, 3, (1, 1), (1, 1),
                                temp_conv, training=self.training, name='conv10')
                temp_conv = ConvMD(2, 128, 128, 3, (1, 1), (1, 1),
                                temp_conv, training=self.training, name='conv11')
                temp_conv = ConvMD(2, 128, 128, 3, (1, 1), (1, 1),
                                temp_conv, training=self.training, name='conv12')
                temp_conv = ConvMD(2, 128, 128, 3, (1, 1), (1, 1),
                                temp_conv, training=self.training, name='conv13')
                deconv2 = Deconv2D(128, 256, 2, (2, 2), (0, 0),
                                temp_conv, training=self.training, name='deconv2')

                # block3:
                temp_conv = ConvMD(2, 128, 256, 3, (2, 2), (1, 1),
                                temp_conv, training=self.training, name='conv14')
                temp_conv = ConvMD(2, 256, 256, 3, (1, 1), (1, 1),
                                temp_conv, training=self.training, name='conv15')
                temp_conv = ConvMD(2, 256, 256, 3, (1, 1), (1, 1),
                                temp_conv, training=self.training, name='conv16')
                temp_conv = ConvMD(2, 256, 256, 3, (1, 1), (1, 1),
                                temp_conv, training=self.training, name='conv17')
                temp_conv = ConvMD(2, 256, 256, 3, (1, 1), (1, 1),
                                temp_conv, training=self.training, name='conv18')
                temp_conv = ConvMD(2, 256, 256, 3, (1, 1), (1, 1),
                                temp_conv, training=self.training, name='conv19')
                deconv3 = Deconv2D(256, 256, 4, (4, 4), (0, 0),
                                temp_conv, training=self.training, name='deconv3')

                # final: 768 = 256*3
                temp_conv = tf.concat([deconv3, deconv2, deconv1], -1)
            elif cfg.RPN_TYPE == 'res_sequeeze':
                from model.squeeze import res_squeeze_net
                temp_conv = res_squeeze_net(temp_conv, self.training)
            elif cfg.RPN_TYPE == 'res_net':
                from model.squeeze import res_net
                temp_conv = res_net(temp_conv, self.training)
            elif cfg.RPN_TYPE == 'avod':
                from model.avod import avod
                temp_conv = avod(temp_conv, self.training)
            elif cfg.RPN_TYPE == 'avod_lite':
                from model.avod import avod_lite
                temp_conv = avod_lite(temp_conv, self.training)
            elif cfg.RPN_TYPE == 'rfbnet':
                from model.rfbnet import RFBNet
                temp_conv = RFBNet(temp_conv, self.training)

            assert temp_conv.get_shape()[1] == cfg.FEATURE_HEIGHT
            assert temp_conv.get_shape()[2] == cfg.FEATURE_WIDTH
            Cout = temp_conv.get_shape()[3]
            self.output_shape = [cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH]

            # Probability score map, scale = [None, FEATURE_HEIGHT, FEATURE_WIDTH, AT]
            p_map = ConvMD(2, Cout, cfg.ANCHOR_TYPES, k=1, s=(1, 1), p=(0, 0),
                           input=temp_conv, training=self.training, activation=False, bn=False, name='conv20')
            # Regression(residual) map, scale = [None, FEATURE_HEIGHT, FEATURE_WIDTH, AT * AL]
            r_map = ConvMD(2, Cout, cfg.ANCHOR_TYPES * cfg.ANCHOR_LEN, 1, (1, 1), (0, 0),
                           temp_conv, training=self.training, activation=False, bn=False, name='conv21')
            # softmax output for positive anchor and negative anchor, scale = [None, FEATURE_HEIGHT, FEATURE_WIDTH, AT]
            # just for one class now, use sigmoid
            self.p_pos = tf.sigmoid(p_map)
            # Direction predict map
            #d_map = ConvMD(2, Cout, cfg.ANCHOR_TYPES * 2, k=1, s=(1, 1), p=(0, 0),
            #               input=temp_conv, training=self.training, activation=False, bn=False, name='conv22')

            # ------- classification loss --------
            if cfg.CLS_LOSS_TYPE == 'focal_loss':
                fl_gamma = 2.0
                fl_alpha = 0.25
                self.cls_pos_loss = -self.pos_equal_one * fl_alpha * tf.pow(1. - self.p_pos, fl_gamma) * tf.log(self.p_pos + small_addon_for_BCE)
                self.cls_neg_loss = -self.neg_equal_one * (1 - fl_alpha) * tf.pow(self.p_pos, fl_gamma) * tf.log(1 - self.p_pos + small_addon_for_BCE)
                self.cls_pos_loss_rec = tf.reduce_sum( self.cls_pos_loss / self.pos_equal_one_sum )
                self.cls_neg_loss_rec = tf.reduce_sum( self.cls_neg_loss / self.neg_equal_one_sum )
                self.cls_loss = self.cls_pos_loss_rec + self.cls_neg_loss_rec
            elif cfg.CLS_LOSS_TYPE == 'voxelnet':
                self.cls_pos_loss = (-self.pos_equal_one * tf.log(self.p_pos + small_addon_for_BCE)) / self.pos_equal_one_sum
                self.cls_neg_loss = (-self.neg_equal_one * tf.log(1 - self.p_pos + small_addon_for_BCE)) / self.neg_equal_one_sum

                self.cls_loss = tf.reduce_sum( alpha * self.cls_pos_loss + beta * self.cls_neg_loss )
                self.cls_pos_loss_rec = tf.reduce_sum( self.cls_pos_loss )
                self.cls_neg_loss_rec = tf.reduce_sum( self.cls_neg_loss )

            def add_sin_difference(box_preds, reg_targets):
                assert cfg.ANCHOR_LEN == 7
                # sin(a - b) = sinacosb-cosasinb
                boxes1 = tf.reshape(box_preds, [-1, 7])
                boxes2 = tf.reshape(reg_targets, [-1, 7])
                rad_pred_encoding = tf.sin(boxes1[..., -1:]) * tf.cos(boxes2[..., -1:])
                rad_gt_encoding = tf.cos(boxes1[..., -1:]) * tf.sin(boxes2[..., -1:])
                boxes1 = tf.concat([boxes1[..., :-1], rad_pred_encoding], axis=-1)
                boxes2 = tf.concat([boxes2[..., :-1], rad_gt_encoding], axis=-1)
                box_preds = tf.reshape(boxes1, [-1, cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH, cfg.ANCHOR_TYPES * cfg.ANCHOR_LEN])
                reg_targets = tf.reshape(boxes2, [-1, cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH, cfg.ANCHOR_TYPES * cfg.ANCHOR_LEN])
                return box_preds, reg_targets
            if cfg.USE_SIN_DIFFERENCE:
                box_preds, reg_targets = add_sin_difference(r_map, self.targets)
            else:
                box_preds, reg_targets = r_map, self.targets

            # -------- regression loss --------
            self.reg_loss = smooth_l1(box_preds * self.pos_equal_one_for_reg, reg_targets * self.pos_equal_one_for_reg, sigma)
            self.reg_loss = self.reg_loss / self.pos_equal_one_sum
            self.reg_loss = tf.reduce_sum(self.reg_loss)

            self.loss = 1 * self.cls_loss + 2 * self.reg_loss

            self.delta_output = r_map
            self.prob_output = self.p_pos


def smooth_l1(deltas, targets, sigma=3.0):
    sigma2 = sigma * sigma
    diffs = tf.subtract(deltas, targets)
    smooth_l1_signs = tf.cast(tf.less(tf.abs(diffs), 1.0 / sigma2), tf.float32)

    smooth_l1_option1 = tf.multiply(diffs, diffs) * 0.5 * sigma2
    smooth_l1_option2 = tf.abs(diffs) - 0.5 / sigma2
    smooth_l1_add = tf.multiply(smooth_l1_option1, smooth_l1_signs) + \
        tf.multiply(smooth_l1_option2, 1 - smooth_l1_signs)
    smooth_l1 = smooth_l1_add

    return smooth_l1


def ConvMD(M, Cin, Cout, k, s, p, input, training, activation=True, bn=True, name='conv'):
    temp_p = np.array(p)
    temp_p = np.lib.pad(temp_p, (1, 1), 'constant', constant_values=(0, 0))
    with tf.variable_scope(name) as scope:
        if(M == 2):
            paddings = (np.array(temp_p)).repeat(2).reshape(4, 2)
            pad = tf.pad(input, paddings, "CONSTANT")
            temp_conv = tf.layers.conv2d(
                pad, Cout, k, strides=s, padding="valid", use_bias=False, reuse=tf.AUTO_REUSE, name=scope)
        if(M == 3):
            paddings = (np.array(temp_p)).repeat(2).reshape(5, 2)
            pad = tf.pad(input, paddings, "CONSTANT")
            temp_conv = tf.layers.conv3d(
                pad, Cout, k, strides=s, padding="valid", use_bias=False, reuse=tf.AUTO_REUSE, name=scope)
        if bn:
            temp_conv = tf.layers.batch_normalization(
                temp_conv, axis=-1, fused=True, training=training, reuse=tf.AUTO_REUSE, name=scope)
        if activation:
            return tf.nn.relu(temp_conv)
        else:
            return temp_conv

def Deconv2D(Cin, Cout, k, s, p, input, training=True, bn=True, name='deconv'):
    temp_p = np.array(p)
    temp_p = np.lib.pad(temp_p, (1, 1), 'constant', constant_values=(0, 0))
    paddings = (np.array(temp_p)).repeat(2).reshape(4, 2)
    pad = tf.pad(input, paddings, "CONSTANT")
    with tf.variable_scope(name) as scope:
        temp_conv = tf.layers.conv2d_transpose(
            pad, Cout, k, strides=s, padding="SAME", use_bias=False, reuse=tf.AUTO_REUSE, name=scope)
        if bn:
            temp_conv = tf.layers.batch_normalization(
                temp_conv, axis=-1, fused=True, training=training, reuse=tf.AUTO_REUSE, name=scope)
        return tf.nn.relu(temp_conv)


if(__name__ == "__main__"):
    m = MiddleAndRPN(tf.placeholder(
        tf.float32, [None, 10, cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH, 128]))
