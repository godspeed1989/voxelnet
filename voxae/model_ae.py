import os, sys
import tensorflow as tf
import numpy as np

if __name__ != '__main__':
    from config import cfg
else:
    from easydict import EasyDict as edict
    cfg = edict()
    cfg.VOXEL_POINT_COUNT = 50
    cfg.VOXEL_X_SIZE = 0.2
    cfg.VOXEL_Y_SIZE = 0.2
    cfg.VOXEL_Z_SIZE = 4

def conv3d(pc_feature, k, s, Cout, scope, training, activation=tf.nn.relu, trainable=True):
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

def ae_encoder(inputs, mask, training, trainable=True):
    with tf.variable_scope('ae_encoder'):
        return None

def ae_decoder(feature, mask, training):
    with tf.variable_scope('ae_decoder'):
        return None

"""
Output: (B,H,W,C)
"""
def py_pc_to_voxel(pc_in, mask_in):
    assert pc_in.shape[0] == mask_in.shape[0]
    assert pc_in.shape[1] == mask_in.shape[1]
    assert pc_in.shape[2] == 4  # with intensity
    assert mask_in.shape[2] == 1
    # X,Y,Z
    voxel_size = np.array([0.05, 0.05, 0.5])
    grid_size = np.array([cfg.VOXEL_X_SIZE / voxel_size[0],
                          cfg.VOXEL_Y_SIZE / voxel_size[1],
                          cfg.VOXEL_Z_SIZE / voxel_size[2]], dtype=np.int32)
    #
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
        voxel_index = np.floor(pc[:,:3] / voxel_size).astype(np.int32)
        #
        bound_x = np.logical_and(voxel_index[:, 0] >= 0, voxel_index[:, 0] < grid_size[0])
        bound_y = np.logical_and(voxel_index[:, 1] >= 0, voxel_index[:, 1] < grid_size[1])
        bound_z = np.logical_and(voxel_index[:, 2] >= 0, voxel_index[:, 2] < grid_size[2])
        bound_box = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)
        pc = pc[bound_box]
        voxel_index = voxel_index[bound_box]
        #
        voxel = np.zeros(grid_size, dtype=np.float32)
        for v, p in zip(voxel_index, pc):
            if voxel[tuple(v)] < p[3]:
                voxel[tuple(v)] = p[3]
        voxel = np.expand_dims(voxel, axis=0)
        voxel = np.expand_dims(voxel, axis=-1)
        voxels.append(voxel)
    return np.concatenate(voxels)

def voxnet_ae(inputs, mask, training):
    #codec = ae_encoder(inputs, mask, training)
    #result = ae_decoder(codec, mask, training)
    voxels = tf.py_func(py_pc_to_voxel, [inputs, mask], tf.float32)
    return voxels

if __name__ == '__main__':
    batch_size = 64
    point_cloud_pl = tf.placeholder(tf.float32, [None, cfg.VOXEL_POINT_COUNT, 4])
    mask_pl = tf.placeholder(tf.bool, [None, cfg.VOXEL_POINT_COUNT, 1])
    training = tf.placeholder(tf.bool)

    result = voxnet_ae(point_cloud_pl, mask_pl, training)
    print(result.shape)

    def gen_batch():
        np.random.seed()
        point_cloud = np.random.rand(batch_size, cfg.VOXEL_POINT_COUNT, 4)
        point_cloud[:,:3] = point_cloud[:,:3] * 3
        mask = np.random.choice(a=[False, True], size=(batch_size, cfg.VOXEL_POINT_COUNT, 1), p=[0.8, 0.2])
        for i in range(batch_size):
            if np.sum(mask[i]) == 0:
                mask[i,np.random.randint(cfg.VOXEL_POINT_COUNT),0] = True
        return point_cloud, mask

    sess = tf.Session()
    point_cloud, mask = gen_batch()
    ret = sess.run(result, {point_cloud_pl: point_cloud, mask_pl: mask, training: False})
    print(ret.shape)
    sys.exit(0)

    loss = tf.reduce_sum(tf.abs(result - point_cloud_pl))
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
