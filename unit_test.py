import os
import fire
import numpy as np

from utils.data_aug import aug_data
from utils.db_sampler import DataBaseSampler
from utils.viewer import view_pc
from utils.utils import label_to_gt_box3d, center_to_corner_box3d

def test_aug_data():
    tag = '%06d' % (12)
    root_path = '/mine/KITTI_DAT/'
    object_dir = os.path.join(root_path, 'training')
    info_path = os.path.join(root_path, 'kitti_dbinfos_train.pkl')
    db_sampler = DataBaseSampler(root_path, info_path)
    tag, rgb, lidar, voxel_dict, label = aug_data(tag, object_dir, sampler=db_sampler)
    print(lidar.shape)
    print(label)
    gt_box3d = label_to_gt_box3d([tag], np.array(label)[np.newaxis, :], cls='', coordinate='lidar')[0]
    gt_box3d_corners = center_to_corner_box3d(gt_box3d)
    view_pc(lidar, gt_box3d_corners)


if __name__ == '__main__':
    fire.Fire()