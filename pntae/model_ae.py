import os, sys
import tensorflow as tf
import numpy as np

if __name__ != '__main__':
    from config import cfg
else:
    from easydict import EasyDict as edict
    cfg = edict()
    cfg.VOXEL_POINT_COUNT = 50

def pointconv(pc_feature, Cout, scope, training, activation=True, trainable=True):
    """
    Input: (B,N,C)
    Return: (B,N,Cout)
    """
    with tf.variable_scope(scope):
        pc_feature = tf.layers.conv1d(pc_feature, Cout, kernel_size=1, strides=1, trainable=trainable,
                                      padding="valid", reuse=tf.AUTO_REUSE, name='conv1d')
        pc_feature = tf.layers.batch_normalization(pc_feature, axis=-1, trainable=trainable,
                        fused=True, training=training, reuse=tf.AUTO_REUSE, name='bn')
        if activation:
            return tf.nn.relu(pc_feature)
        else:
            return pc_feature

def pointdeconv(pc_feature, Cout, scope, kernel_size, strides, training, activation=True):
    """
    Input: (B,H,W,C)
    Return: (B,H,W,Cout)
    """
    with tf.variable_scope(scope):
        pc_feature = tf.layers.conv2d_transpose(pc_feature, Cout, kernel_size=kernel_size, strides=strides,
                        padding="valid", reuse=tf.AUTO_REUSE, name='deconv2d')
        pc_feature = tf.layers.batch_normalization(pc_feature, axis=-1,
                        fused=True, training=training, reuse=tf.AUTO_REUSE, name='bn')
        if activation:
            return tf.nn.relu(pc_feature)
        else:
            return pc_feature

"""
Input: (B,N,C)
Return: (B,Cout)
"""
def ae_encoder(inputs, mask, training, trainable=True):
    with tf.variable_scope('ae_encoder'):
        pc_feature = pointconv(inputs, 32, 'encoder_conv1', training, trainable=trainable)
        pc_feature = pointconv(pc_feature, 64, 'encoder_conv2', training, trainable=trainable)
        # [K, T, 1] -> [K, T, 64]
        mask64 = tf.tile(mask, [1, 1, 64])
        pc_feature = tf.multiply(pc_feature, tf.cast(mask64, tf.float32))

        feature = tf.concat([inputs, pc_feature], axis=-1)
        feature = pointconv(feature, 128, 'encoder_conv3', training, trainable=trainable)
        feature = pointconv(feature, 256, 'encoder_conv4', training, trainable=trainable)

        # [K, T, 1] -> [K, T, 256]
        mask256 = tf.tile(mask, [1, 1, 256])
        feature = tf.multiply(feature, tf.cast(mask256, tf.float32))
        # [K, T, 256] -> [K, 256]
        aggregated = tf.reduce_max(feature, axis=1, keepdims=False)

        pointsize = tf.reduce_sum(tf.cast(mask, tf.float32), axis=1, keepdims=False)
        aggregated = tf.concat([aggregated, pointsize], axis=-1)

        aggregated = tf.layers.dense(aggregated, 256, activation=tf.nn.relu, trainable=trainable)

        return aggregated

"""
Input: (B,Cout)
Return: (B,N,C)
"""
def ae_decoder(feature, mask, training):
    with tf.variable_scope('ae_decoder'):
        feature = tf.expand_dims(feature, axis=1)
        feature = tf.expand_dims(feature, axis=1)
        # 1x1 -> 2x2
        deconv = pointdeconv(feature, 128, 'decoder_deconv1',
                            kernel_size=[2,2], strides=[1,1], training=training)
        # 2x2 -> 4x4
        deconv = pointdeconv(deconv, 64, 'decoder_deconv2',
                            kernel_size=[3,3], strides=[1,1], training=training)
        # 4x4 -> 5x10
        #deconv = pointdeconv(deconv, 3, 'decoder_deconv3',
        #                    kernel_size=[2,7], strides=[1,1], training=training)
        deconv = tf.layers.conv2d(deconv, 3, kernel_size=1, strides=1,
                                    padding="valid", reuse=tf.AUTO_REUSE, name='decoder_deconv3')
        deconv = tf.reshape(deconv, [-1, cfg.VOXEL_POINT_COUNT, 3])
        return deconv

def pntnet_ae(inputs, mask, training):
    codec = ae_encoder(inputs, mask, training)
    result = ae_decoder(codec, mask, training)
    return result

if __name__ == '__main__':
    batch_size = 64
    point_cloud_pl = tf.placeholder(tf.float32, [None, cfg.VOXEL_POINT_COUNT, 3])
    mask_pl = tf.placeholder(tf.bool, [None, cfg.VOXEL_POINT_COUNT, 1])
    training = tf.placeholder(tf.bool)

    result = pntnet_ae(point_cloud_pl, mask_pl, training)
    assert result.get_shape()[1].value == cfg.VOXEL_POINT_COUNT
    assert result.get_shape()[2].value == 3

    def gen_batch():
        np.random.seed()
        point_cloud = np.random.rand(batch_size, cfg.VOXEL_POINT_COUNT, 3)
        mask = np.random.choice(a=[False, True], size=(batch_size, cfg.VOXEL_POINT_COUNT, 1), p=[0.8, 0.2])
        for i in range(batch_size):
            if np.sum(mask[i]) == 0:
                mask[i,np.random.randint(cfg.VOXEL_POINT_COUNT),0] = True
        return point_cloud, mask

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(BASE_DIR, 'nn_distance'))
    from tf_nndistance import nn_distance
    dists_forward, _, dists_backward, _ = nn_distance(result, point_cloud_pl)
    loss_pred = tf.reduce_mean(dists_forward + dists_backward)
    tf.summary.scalar('loss_pred', loss_pred)
    if False:
        tvars = tf.trainable_variables()
        lossL1 = tf.add_n([ tf.reduce_sum(tf.abs(v)) for v in tvars if 'bias' not in v.name]) * 0.001
        tf.summary.scalar('lossL1', lossL1)
        loss = loss_pred + lossL1
    else:
        loss = loss_pred
    train = tf.train.AdamOptimizer(learning_rate=0.00005).minimize(loss)

    saver = tf.train.Saver(max_to_keep=2)

    best_loss, step = 1e20, 0
    with tf.Session() as sess:
        summary = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter('.', sess.graph)
        for i in range(100000):
            point_cloud, mask = gen_batch()
            loss_val, _, summary_val = sess.run([loss, train, summary], {point_cloud_pl: point_cloud, mask_pl: mask, training: True})
            train_writer.add_summary(summary_val, i)
            if i and i % 100 == 0:
                step = step + 1
                print(step, loss_val)
                if loss_val < best_loss:
                    best_loss = loss_val
                    save_path = saver.save(sess, './model.%.6f.ckpt' % (loss_val))
