import os, sys
import tensorflow as tf
import numpy as np

if __name__ != '__main__':
    from config import cfg
else:
    from easydict import EasyDict as edict
    cfg = edict()
    cfg.VOXEL_POINT_COUNT = 64
    cfg.VOXEL_X_SIZE = 0.2
    cfg.VOXEL_Y_SIZE = 0.2
    cfg.VOXEL_Z_SIZE = 4
    cfg.VOXVOX_SIZE = np.array([0.05, 0.05, 0.5], dtype=np.float32)
    # (4, 4, 8)
    cfg.VOXVOX_GRID_SIZE = np.array([cfg.VOXEL_X_SIZE // (cfg.VOXVOX_SIZE[0] - 1e-6),
                                     cfg.VOXEL_Y_SIZE // (cfg.VOXVOX_SIZE[1] - 1e-6),
                                     cfg.VOXEL_Z_SIZE // (cfg.VOXVOX_SIZE[2] - 1e-6)], dtype=np.int32)

def conv3d(pc_feature, k, s, Cout, scope, training, trainable=True, activation=tf.nn.relu):
    """
    Input: (B,H,W,D,C)
    Return: (B,H,W,D,Cout)
    """
    with tf.variable_scope(scope):
        pc_feature = tf.layers.conv3d(pc_feature, Cout, kernel_size=k, strides=s, trainable=trainable,
                                      padding="valid", reuse=tf.AUTO_REUSE, name='conv3d')
        pc_feature = tf.layers.batch_normalization(pc_feature, axis=-1, trainable=trainable,
                        fused=True, training=training, reuse=tf.AUTO_REUSE, name='bn')
        if activation:
            return activation(pc_feature)
        else:
            return pc_feature

def deconv3d(pc_feature, k, s, Cout, scope, training, activation=tf.nn.relu):
    """
    Input: (B,H,W,C)
    Return: (B,H,W,Cout)
    """
    with tf.variable_scope(scope):
        pc_feature = tf.layers.conv3d_transpose(pc_feature, Cout, kernel_size=k, strides=s,
                        padding="valid", reuse=tf.AUTO_REUSE, name='deconv3d')
        pc_feature = tf.layers.batch_normalization(pc_feature, axis=-1,
                        fused=True, training=training, reuse=tf.AUTO_REUSE, name='bn')
        if activation:
            return activation(pc_feature)
        else:
            return pc_feature

def ae_encoder(inputs, training, trainable=True):
    with tf.variable_scope('vae_encoder'):
        vox_feature = conv3d(inputs, 2, 1, 16, 'encoder_conv1', training, trainable)
        vox_feature = conv3d(vox_feature, 2, 1, 32, 'encoder_conv2', training, trainable)
        vox_feature = conv3d(vox_feature, 2, 1, 64, 'encoder_conv3', training, trainable)
        vox_feature = tf.squeeze(vox_feature, axis=[1,2])
        #vox_feature = tf.layers.flatten(vox_feature)
        vox_feature = tf.reduce_max(vox_feature, axis=1, keepdims=False)
        print(vox_feature.shape)
        vox_feature = tf.layers.dense(vox_feature, 64, activation=tf.nn.relu, name='fc', trainable=trainable)
        '''
        vox_feature = tf.reshape(inputs, [-1, np.product(tuple(cfg.VOXVOX_GRID_SIZE))])
        vox_feature = tf.layers.dense(vox_feature, 128, activation=tf.nn.relu, name='fc1', trainable=trainable)
        vox_feature = tf.nn.dropout(vox_feature, keep_prob=0.5)
        vox_feature = tf.layers.dense(vox_feature, 64, activation=tf.nn.relu, name='fc2', trainable=trainable)
        '''
        return vox_feature

def ae_decoder(feature, training):
    with tf.variable_scope('vae_decoder'):
        feature = tf.layers.dense(feature, 128, activation=tf.nn.relu, name='fc1')
        feature = tf.nn.dropout(feature, keep_prob=0.5)
        feature = tf.layers.dense(feature, 256, activation=tf.nn.relu, name='fc2')
        feature = tf.nn.dropout(feature, keep_prob=0.5)
        feature = tf.layers.dense(feature, np.product(cfg.VOXVOX_GRID_SIZE),
                                  activation=tf.nn.sigmoid, name='fc_out')
        feature = tf.reshape(feature, [-1, *tuple(cfg.VOXVOX_GRID_SIZE), 1])
        return feature

"""
Output: (B,H,W,C)
"""
def py_pc_to_voxel(pc_in, mask_in):
    assert pc_in.shape[0] == mask_in.shape[0]
    assert pc_in.shape[1] == mask_in.shape[1]
    assert pc_in.shape[2] == 4  # with intensity
    assert mask_in.shape[2] == 1
    # X,Y,Z
    batch_size = pc_in.shape[0]
    voxels = []
    for i in range(batch_size):
        pc, mask = pc_in[i], mask_in[i]
        pc = pc[np.squeeze(mask)]
        assert pc.shape[0]
        #
        min_box_coor = np.min(pc, axis=0)
        min_box_coor[3] = 0
        pc = pc - min_box_coor
        voxel_index = np.floor(pc[:,:3] / cfg.VOXVOX_SIZE).astype(np.int32)
        #
        bound_x = np.logical_and(voxel_index[:, 0] >= 0, voxel_index[:, 0] < cfg.VOXVOX_GRID_SIZE[0])
        bound_y = np.logical_and(voxel_index[:, 1] >= 0, voxel_index[:, 1] < cfg.VOXVOX_GRID_SIZE[1])
        bound_z = np.logical_and(voxel_index[:, 2] >= 0, voxel_index[:, 2] < cfg.VOXVOX_GRID_SIZE[2])
        bound_box = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)
        pc = pc[bound_box]
        voxel_index = voxel_index[bound_box]
        #
        voxel = np.zeros(cfg.VOXVOX_GRID_SIZE, dtype=np.float32)
        for v, p in zip(voxel_index, pc):
            voxel[tuple(v)] = 1
        voxel = np.expand_dims(voxel, axis=0)
        voxel = np.expand_dims(voxel, axis=-1)
        voxels.append(voxel)
    return np.concatenate(voxels)

def pc_to_voxel(inputs, mask):
    voxels = tf.py_func(py_pc_to_voxel, [inputs, mask], tf.float32)
    HWD = tuple(cfg.VOXVOX_GRID_SIZE)
    voxels.set_shape([None, *HWD, 1])
    return voxels

def voxnet_ae(inputs, mask, training):
    voxels = pc_to_voxel(inputs, mask)
    #
    codec = ae_encoder(voxels, training)
    result = ae_decoder(codec, training)
    return result, voxels

'''
# from https://github.com/ajbrock/Generative-and-Discriminative-Voxel-Modeling/blob/master/Generative/train_VAE.py
# Weighted binary cross-entropy for use in voxel loss. Allows weighting of false positives relative to false negatives.
# Nominally set to strongly penalize false negatives
def weighted_binary_crossentropy(output,target):
    return -(98.0*target * T.log(output) + 2.0*(1.0 - target) * T.log(1.0 - output))/100.0

# Voxel-Wise Reconstruction Loss
# Note that the output values are clipped to prevent the BCE from evaluating log(0).
voxel_loss = T.cast(T.mean(weighted_binary_crossentropy(T.clip(lasagne.nonlinearities.sigmoid( X_hat ), 1e-7, 1.0 - 1e-7), X)),'float32')
'''
def voxel_loss(output, target):
    output = tf.clip_by_value(output, 1e-7, 1.0 - 1e-7)
    weighted_binary_crossentropy = -(98.0*target * tf.log(output) + 2.0*(1.0 - target) * tf.log(1.0 - output))
    return tf.reduce_mean(weighted_binary_crossentropy)

if __name__ == '__main__':
    batch_size = 64
    point_cloud_pl = tf.placeholder(tf.float32, [None, cfg.VOXEL_POINT_COUNT, 4])
    mask_pl = tf.placeholder(tf.bool, [None, cfg.VOXEL_POINT_COUNT, 1])
    training = tf.placeholder(tf.bool)

    result, voxels = voxnet_ae(point_cloud_pl, mask_pl, training)
    print(result.shape)

    def gen_batch():
        np.random.seed()
        point_cloud = np.random.rand(batch_size, cfg.VOXEL_POINT_COUNT, 4)
        point_cloud[:,:3] = point_cloud[:,:3] * 4
        mask = np.random.choice(a=[False, True], size=(batch_size, cfg.VOXEL_POINT_COUNT, 1), p=[0.2, 0.8])
        for i in range(batch_size):
            if np.sum(mask[i]) == 0:
                mask[i,np.random.randint(cfg.VOXEL_POINT_COUNT),0] = True
        return point_cloud, mask
    '''
    sess = tf.Session()
    point_cloud, mask = gen_batch()
    sess.run(tf.global_variables_initializer())
    pred, gt = sess.run([result, voxels], {point_cloud_pl: point_cloud, mask_pl: mask, training: False})
    print(pred.shape, gt.shape)
    sys.exit(0)
    '''

    #loss_pred = tf.reduce_sum(tf.abs(result - voxels))
    loss_pred = voxel_loss(result, voxels)
    tf.summary.scalar('loss_pred', loss_pred)
    train = tf.train.AdamOptimizer(learning_rate=0.00005).minimize(loss_pred)

    saver = tf.train.Saver(max_to_keep=2)

    best_loss, step = 1e20, 0
    with tf.Session() as sess:
        summary = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter('.', sess.graph)
        for i in range(100000):
            point_cloud, mask = gen_batch()
            loss_val, _, summary_val = sess.run([loss_pred, train, summary],
                {point_cloud_pl: point_cloud, mask_pl: mask, training: True})
            train_writer.add_summary(summary_val, i)
            if i and i % 100 == 0:
                step = step + 1
                print(step, loss_val)
                if loss_val < best_loss:
                    best_loss = loss_val
                    save_path = saver.save(sess, './model.%.6f.ckpt' % (loss_val))
