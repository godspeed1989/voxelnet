
import numpy as np
import os, glob
import tensorflow as tf

from utils.preprocess import process_pointcloud
from pntae.model_ae import pntnet_ae
from pntae.nn_distance.tf_nndistance import nn_distance
from config import cfg

batch_size = 64

def gen_batch(f_lidar):
    np.random.seed()
    lidar = np.random.rand(12345, 4)
    # raw_lidar = np.fromfile(self.f_lidar, dtype=np.float32).reshape((-1, 4))

    voxel_dict = process_pointcloud('pc', lidar)
    num_voxels = voxel_dict['feature_buffer'].shape[0]
    if num_voxels < batch_size:
        return None, None
    index = np.random.choice(num_voxels, batch_size, replace=False)
    batch_pc = voxel_dict['feature_buffer'][index, :3]
    batch_mask = voxel_dict['mask_buffer'][index]
    return batch_pc, batch_mask

def train():
    point_cloud_pl = tf.placeholder(tf.float32, [None, cfg.VOXEL_POINT_COUNT, 3])
    mask_pl = tf.placeholder(tf.bool, [None, cfg.VOXEL_POINT_COUNT, 1])
    training = tf.placeholder(tf.bool)

    result = pntnet_ae(point_cloud_pl, mask_pl, training)
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

    sess = tf.Session()
    summary = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./pntae/', sess.graph)

    # get lidar file list
    train_dir = os.path.join(cfg.DATA_DIR, 'training')
    f_lidar = glob.glob(os.path.join(train_dir, 'velodyne', '*.bin'))

    best_loss, step = 1e20, 0
    for epoch in range(2):
        for f in range(3):
            point_cloud, mask = gen_batch(f)
            if point_cloud is None:
                continue
            step = step + 1
            loss_val, _, summary_val = sess.run([loss, train, summary],
                {point_cloud_pl: point_cloud, mask_pl: mask, training: True})
            train_writer.add_summary(summary_val, step)
            if step and step % 100 == 0:
                print(step, loss_val)
                if loss_val < best_loss:
                    best_loss = loss_val
                    saver.save(sess, './pntae/model.%.6f.ckpt' % (loss_val))

if __name__ == '__main__':
    train()