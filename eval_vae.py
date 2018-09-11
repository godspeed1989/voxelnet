import numpy as np
import os, sys, glob
import tensorflow as tf

from utils.preprocess import process_pointcloud, extract_lidar_in_fov
from voxae.model_ae import voxnet_ae
from config import cfg
from train_vae import batch_size

class eval_AE(object):
    def __init__(self):
        self.point_cloud_pl = tf.placeholder(tf.float32, [None, cfg.VOXEL_POINT_COUNT, 4])
        self.mask_pl = tf.placeholder(tf.bool, [None, cfg.VOXEL_POINT_COUNT, 1])
        self.training = tf.placeholder(tf.bool)
        self.result, _ = voxnet_ae(self.point_cloud_pl, self.mask_pl, self.training)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, tf.train.latest_checkpoint('./voxae'))

    def eval(self, point_cloud, mask):
        pred = self.sess.run(self.result, {self.point_cloud_pl: point_cloud,
                                           self.mask_pl: mask, self.training: False})
        return pred

def write_ply(fname, pc):
    print('write ply', fname)
    fout = open(fname, 'w')
    fout.write('ply\n')
    fout.write('format   ascii   1.0\n')
    fout.write('element   vertex  {}\n'.format(pc.shape[0]))
    fout.write('property   float32   x\n')
    fout.write('property   float32   y\n')
    fout.write('property   float32   z\n')
    fout.write('end_header\n')
    for p in pc:
        fout.write('{} {} {}\n'.format(p[0], p[1], p[2]))
    fout.close()

if __name__ == '__main__':
    eae = eval_AE()

    val_dir = os.path.join(cfg.DATA_DIR, 'training')
    f_lidar = glob.glob(os.path.join(val_dir, 'velodyne', '*.bin'))
    print('Total {} file'.format(len(f_lidar)))
    c = np.random.randint(len(f_lidar))
    print('choose {}'.format(f_lidar[c]))
    lidar = np.fromfile(f_lidar[c], dtype=np.float32).reshape((-1, 4))
    if cfg.FOV_FILTER:
        lidar = extract_lidar_in_fov(lidar)

    voxel_dict = process_pointcloud('pc', lidar)
    num_voxels = voxel_dict['feature_buffer'].shape[0]
    assert num_voxels == voxel_dict['coordinate_buffer'].shape[0]
    print('num_voxels', num_voxels)

    # generate mesh grid
    linx = np.linspace(0, cfg.VOXEL_X_SIZE, cfg.VOXVOX_GRID_SIZE[0])
    liny = np.linspace(0, cfg.VOXEL_Y_SIZE, cfg.VOXVOX_GRID_SIZE[1])
    linz = np.linspace(0, cfg.VOXEL_Z_SIZE, cfg.VOXVOX_GRID_SIZE[2])
    mesh = np.meshgrid(linx, liny, linz)
    linx = np.expand_dims(mesh[0], axis=-1) # (4, 4, 8, 1)
    liny = np.expand_dims(mesh[1], axis=-1)
    linz = np.expand_dims(mesh[2], axis=-1)
    mesh_coord = np.concatenate([linx, liny, linz], axis=-1)
    print('mesh_coord', mesh_coord.shape)

    it = num_voxels // batch_size
    results = []
    for i in range(it):
        batch_pc = voxel_dict['feature_buffer'][i*batch_size:(i+1)*batch_size]
        batch_mask = voxel_dict['mask_buffer'][i*batch_size:(i+1)*batch_size]
        # voxel to pc
        ret = eae.eval(batch_pc, batch_mask)
        #print(ret)
        voxel_indice = voxel_dict['coordinate_buffer'][i*batch_size:(i+1)*batch_size]
        for j in range(batch_size):
            choice = ret[j] > 0.5
            indices = choice.astype(np.float32)
            # translation
            vox_idx = voxel_indice[j][::-1] # (Z, Y, X) -> (X, Y, Z)
            points = indices*mesh_coord + \
                     vox_idx*[cfg.VOXEL_X_SIZE, cfg.VOXEL_Y_SIZE, cfg.VOXEL_Z_SIZE]
            choice = np.squeeze(choice, axis=-1)
            points = points[choice]
            results.append(points)

    results = np.concatenate(results)
    print(results.shape)
    print(lidar.shape)
    write_ply('{}_eval_ae.ply'.format(c), results)
    write_ply('{}_eval_orig.ply'.format(c), lidar)
