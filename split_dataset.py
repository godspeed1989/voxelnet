import os, sys
import numpy as np
import glob
import argparse
from shutil import copy

parser = argparse.ArgumentParser(description='split_dataset')
parser.add_argument('--input-path', type=str, nargs='?',
                    default='./trainval', help='dataset input dir')
parser.add_argument('--training-ratio', type=int, nargs='?',
                    default=90, help='training split ratio')
args = parser.parse_args()

def mkdirs(base):
    os.makedirs(base, exist_ok=True)
    label_dir = os.path.join(base, 'label_2')
    image_dir = os.path.join(base, 'image_2')
    lidar_dir = os.path.join(base, 'velodyne')
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(lidar_dir, exist_ok=True)
    return label_dir, image_dir, lidar_dir

def do_copy(label_dir, image_dir, lidar_dir, input_path, name):
    copy(os.path.join(input_path, 'label_2', name+'.txt'), label_dir)
    copy(os.path.join(input_path, 'image_2', name+'.png'), image_dir)
    copy(os.path.join(input_path, 'velodyne', name+'.bin'), lidar_dir)

def split(data_dir, train_ratio):
    f_input = glob.glob(os.path.join(args.input_path, 'label_2', '*.txt'))
    f_input.sort()
    data_tag = []
    for fin in f_input:
        data_tag.append(fin.split('/')[-1].split('.')[-2])
    x = np.array(data_tag)
    print(x.shape[0])
    # split
    training_cnt = (int)(x.shape[0] * args.training_ratio // 100)
    val_cnt = x.shape[0] - training_cnt
    print('training {} validation {}'.format(training_cnt, val_cnt))
    indices = np.random.permutation(x.shape[0])
    training_idx, val_idx = indices[:training_cnt], indices[training_cnt:]
    training, val = x[training_idx], x[val_idx]
    # copy training
    print('copy training...')
    training_dir = os.path.join(os.path.dirname(args.input_path), 'training{}'.format(args.training_ratio))
    label_dir, image_dir, lidar_dir = mkdirs(training_dir)
    for i in range(training_cnt):
        do_copy(label_dir, image_dir, lidar_dir, args.input_path, training[i])
    # copy validation
    print('copy validation...')
    val_dir = os.path.join(os.path.dirname(args.input_path), 'validation{}'.format(100-args.training_ratio))
    label_dir, image_dir, lidar_dir = mkdirs(val_dir)
    for i in range(val_cnt):
        do_copy(label_dir, image_dir, lidar_dir, args.input_path, val[i])
    print('Done')

if __name__ == '__main__':
    print('IN', args.input_path)
    print('TRAIN', args.training_ratio)
    split(args.input_path, args.training_ratio)
