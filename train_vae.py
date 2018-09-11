
import numpy as np
import os, glob
import tensorflow as tf

from utils.preprocess import process_pointcloud
from utils.colorize import colorize
from voxae.model_ae import voxnet_ae, voxel_loss
from config import cfg

batch_size = 32

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

def batch_to_img(b):
    b = b.astype(np.float32)
    b = np.sum(b, axis=3, keepdims=False)
    b = np.reshape(b, [batch_size, cfg.VOXVOX_GRID_SIZE[0]*cfg.VOXVOX_GRID_SIZE[1], 1])
    b = np.transpose(b, axes=[1,0,2])
    return colorize(b)[np.newaxis, ...]

def train():
    point_cloud_pl = tf.placeholder(tf.float32, [None, cfg.VOXEL_POINT_COUNT, 4])
    mask_pl = tf.placeholder(tf.bool, [None, cfg.VOXEL_POINT_COUNT, 1])
    training = tf.placeholder(tf.bool)

    result, voxels = voxnet_ae(point_cloud_pl, mask_pl, training)
    loss_pred = voxel_loss(result, voxels)

    pred_summary = tf.summary.merge([
        tf.summary.scalar('loss_pred', loss_pred)
    ])
    train = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(loss_pred)

    pred_img_pl = tf.placeholder(tf.uint8, [None, cfg.VOXVOX_GRID_SIZE[0]*cfg.VOXVOX_GRID_SIZE[1], batch_size, 3])
    vox_img_pl = tf.placeholder(tf.uint8, [None, cfg.VOXVOX_GRID_SIZE[0]*cfg.VOXVOX_GRID_SIZE[1], batch_size, 3])
    img_summary = tf.summary.merge([
        tf.summary.image('pred_img', pred_img_pl),
        tf.summary.image('vox_img', vox_img_pl)
    ])
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter('./voxae/', sess.graph)
    saver = tf.train.Saver(max_to_keep=2)

    # get lidar file list
    train_dir = os.path.join(cfg.DATA_DIR, 'training')
    f_lidar = glob.glob(os.path.join(train_dir, 'velodyne', '*.bin'))
    print('Total {} file'.format(len(f_lidar)))

    best_loss, step = 1e20, 0
    for epoch in range(10):
        for f in f_lidar:
            point_cloud, mask = gen_batch(f)
            if point_cloud is None:
                continue
            step = step + 1

            loss_val, _, pred_summary_val, pred, voxel = sess.run([loss_pred, train, pred_summary, result, voxels],
                {point_cloud_pl: point_cloud, mask_pl: mask, training: True})
            train_writer.add_summary(pred_summary_val, step)

            pred = pred > 0.5
            img_summary_val = sess.run(img_summary,
                {pred_img_pl: batch_to_img(pred), vox_img_pl: batch_to_img(voxel)})
            train_writer.add_summary(img_summary_val, step)

            if step and step % 100 == 0:
                print(step, loss_val)
                if loss_val < best_loss:
                    best_loss = loss_val
                    saver.save(sess, './voxae/model.%.6f.ckpt' % (loss_val))

if __name__ == '__main__':
    train()