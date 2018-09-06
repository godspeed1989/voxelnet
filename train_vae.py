
import numpy as np
import os, glob
import tensorflow as tf

from utils.preprocess import process_pointcloud
from voxae.model_ae import voxnet_ae
from config import cfg

batch_size = 64

def gen_batch(f_lidar):
    lidar = np.fromfile(f_lidar, dtype=np.float32).reshape((-1, 4))

    voxel_dict = process_pointcloud('pc', lidar)
    num_voxels = voxel_dict['feature_buffer'].shape[0]
    if num_voxels < batch_size:
        return None, None
    index = np.random.choice(num_voxels, batch_size, replace=False)
    batch_pc = voxel_dict['feature_buffer'][index]
    batch_mask = voxel_dict['mask_buffer'][index]
    return batch_pc, batch_mask

def train():
    point_cloud_pl = tf.placeholder(tf.float32, [None, cfg.VOXEL_POINT_COUNT, 4])
    mask_pl = tf.placeholder(tf.bool, [None, cfg.VOXEL_POINT_COUNT, 1])
    training = tf.placeholder(tf.bool)

    result, voxels = voxnet_ae(point_cloud_pl, mask_pl, training)
    if True:
        loss_pred = tf.abs(result - voxels)
        nonzero_sum = tf.reduce_sum(tf.cast(mask_pl, tf.float32), axis=1, keepdims=True)
        nonzero_sum = tf.expand_dims(nonzero_sum, axis=-1)
        nonzero_sum = tf.expand_dims(nonzero_sum, axis=-1)
        nonzero_sum = np.product(cfg.VOXVOX_GRID_SIZE) - nonzero_sum
        loss_pred = loss_pred / nonzero_sum
        loss_pred = tf.reduce_sum(loss_pred)
    tf.summary.scalar('loss_pred', loss_pred)
    train = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(loss_pred)

    sess = tf.Session()
    summary = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter('./voxae/', sess.graph)
    saver = tf.train.Saver(max_to_keep=2)

    # get lidar file list
    train_dir = os.path.join(cfg.DATA_DIR, 'training')
    f_lidar = glob.glob(os.path.join(train_dir, 'velodyne', '*.bin'))
    print('Total {} file'.format(len(f_lidar)))

    best_loss, step = 1e20, 0
    for epoch in range(2):
        for f in f_lidar:
            point_cloud, mask = gen_batch(f)
            if point_cloud is None:
                continue
            step = step + 1
            loss_val, _, pred_summary_val = sess.run([loss_pred, train, summary],
                {point_cloud_pl: point_cloud, mask_pl: mask, training: True})
            train_writer.add_summary(pred_summary_val, step)
            if step and step % 100 == 0:
                print(step, loss_val)
                if loss_val < best_loss:
                    best_loss = loss_val
                    saver.save(sess, './voxae/model.%.6f.ckpt' % (loss_val))

if __name__ == '__main__':
    train()