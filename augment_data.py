import os
import numpy as np
import cv2
import multiprocessing as mp
import argparse
import glob

from config import cfg
from utils.data_aug import aug_data

object_dir = '/mine/KITTI_DAT/validation'
output_dir = cfg.AUG_DATA_FOLDER

def worker(tag):
    new_tag, rgb, lidar, voxel_dict, label = aug_data(tag, object_dir)
    output_path = os.path.join(object_dir, output_dir)

    cv2.imwrite(os.path.join(output_path, 'image_2', new_tag + '.png'), rgb)
    lidar.reshape(-1).tofile(os.path.join(output_path, 'velodyne', new_tag + '.bin'))
    np.savez_compressed(os.path.join(
        output_path, 'voxel' if cfg.DETECT_OBJ == 'Car' else 'voxel_ped', new_tag), **voxel_dict)
    with open(os.path.join(output_path, 'label_2', new_tag + '.txt'), 'w+') as f:
        for line in label:
            f.write(line)

def main(args):
    fl = glob.glob(os.path.join(object_dir, 'label_2', '*.txt'))
    candidate = [f.split('/')[-1].split('.')[0] for f in fl]
    print('generate {} tags'.format(len(candidate)))
    os.makedirs(os.path.join(object_dir, output_dir), exist_ok=True)
    os.makedirs(os.path.join(object_dir, output_dir, 'image_2'), exist_ok=True)
    os.makedirs(os.path.join(object_dir, output_dir, 'velodyne'), exist_ok=True)
    os.makedirs(os.path.join(object_dir, output_dir, 'voxel'), exist_ok=True)
    os.makedirs(os.path.join(object_dir, output_dir, 'label_2'), exist_ok=True)

    pool = mp.Pool(args.num_workers)
    pool.map(worker, candidate)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-n', '--num-workers', type=int, default=6)
    args = parser.parse_args()

    main(args)
