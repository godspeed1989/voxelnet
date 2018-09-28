import os, sys
import glob
import numpy as np
import argparse
import cv2

from config import cfg
from utils.utils import label_to_gt_box3d, load_calib, box3d_to_label
from utils.rotbox_cuda.rbbox_overlaps import py_rbbox_overlaps_3d

from utils.utils import lidar_to_bird_view_img, draw_lidar_box3d_on_birdview

parser = argparse.ArgumentParser(description='training')
parser.add_argument('--input-path', type=str, nargs='?',
                    default='./predictions/79/data', help='results input dir')
parser.add_argument('--output-dir-name', type=str, nargs='?',
                    default='data_calib', help='results output dir')
args = parser.parse_args()

# calibrate the z axis of validaton 

def calib_file(fin, out_dir):
    data_tag = fin.split('/')[-1].split('.')[-2]
    print(data_tag)
    # predict
    labels = [line.rpartition(' ')[0] for line in open(fin, 'r').readlines()] # skip score
    pred_boxes3d = label_to_gt_box3d([data_tag], [labels], cls='Car', coordinate='lidar')[0]
    pred_boxes3d = np.array(pred_boxes3d)
    #print(pred_boxes3d)
    # ground truth
    val_dir = os.path.join(cfg.DATA_DIR, 'validation')
    f_gt = os.path.join(val_dir, 'label_2', data_tag + '.txt')
    gt_labels = [line for line in open(f_gt, 'r').readlines()]
    gt_boxes3d = label_to_gt_box3d([data_tag], [gt_labels], cls='Car', coordinate='lidar')[0]
    gt_boxes3d = np.array(gt_boxes3d)
    #print(gt_boxes3d)
    # load
    P, Tr, R = load_calib( os.path.join( cfg.CALIB_DIR, data_tag + '.txt' ) )
    # calibrate z if the iou with ground truth > 0.5
    if pred_boxes3d.shape[0] and gt_boxes3d.shape[0]:
        iou = py_rbbox_overlaps_3d(np.ascontiguousarray(pred_boxes3d, dtype=np.float32),
                                   np.ascontiguousarray(gt_boxes3d, dtype=np.float32))
        #print(iou)
        for i in range(iou.shape[0]):
            idx = np.argmax(iou[i])
            if iou[i][idx] < 0.5: # find corresponding gt
                continue
            # !!! HERE 1/2
            # x(0) y(1) z(2) h(3) w(4) l(5) r(6)
            pred_boxes3d[i][2:4] = gt_boxes3d[idx][2:4]
        # write calibrated result
        of_path = os.path.join(out_dir, data_tag + '.txt')
        fout = open(of_path, 'w')
        fin_data = open(fin, 'r').readlines()
        assert pred_boxes3d.shape[0] == len(fin_data)
        num_objs = pred_boxes3d.shape[0]
        for i, line in zip(range(num_objs), fin_data):
            ret = line.split()
            #print(ret)
            label = box3d_to_label([pred_boxes3d[np.newaxis, i]], [np.zeros(num_objs)], [np.ones(num_objs)],
                                   coordinate='lidar', P2=P, T_VELO_2_CAM=Tr, R_RECT_0=R)[0][0]
            label = label.split()
            #print(label)
            # !!! HERE 2/2
            # ..., h(-8), w(-7), l(-6), x(-5), y(-4), z(-3), r(-2), score(-1)
            ret[-8] = label[-8]
            ret[-3] = label[-3]
            fout.write(' '.join(ret) + '\n')
        '''
        f_lidar = os.path.join(val_dir, 'velodyne', data_tag + '.bin')
        lidar = np.fromfile(f_lidar, dtype=np.float32).reshape((-1, 4))
        bird_view = lidar_to_bird_view_img(lidar, factor=1)
        bev = draw_lidar_box3d_on_birdview(bird_view, pred_boxes3d, None, gt_boxes3d)
        cv2.imwrite('bev.png', bev)
        assert 0
        '''

def calib():
    print('IN', args.input_path)
    output_path = os.path.join(os.path.dirname(args.input_path), args.output_dir_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print('OUT', output_path)
    # list input files
    f_input = glob.glob(os.path.join(args.input_path, '*.txt'))
    f_input.sort()
    print('NUM FILES', len(f_input))
    for f in f_input:
        calib_file(f, output_path)
    #calib_file(f_input[4], output_path)

if __name__ == '__main__':
    calib()
