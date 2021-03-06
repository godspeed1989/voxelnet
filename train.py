#!/usr/bin/env python
# -*- coding:UTF-8 -*-
import os
import time
import sys
import argparse
import cv2
import numpy as np
import tensorflow as tf
from itertools import count

from config import cfg
from model.model import RPN3D
from utils.kitti_loader import iterate_data, sample_test_data
from utils.utils import box3d_to_label, load_calib
from utils.db_sampler import DataBaseSampler
from train_hook import check_if_should_pause
from termcolor import cprint
import warnings

log_f = None
def log_print(s, color='green', write=True):
    cprint(s, color, attrs=['bold'])
    if write:
        log_f.write(s + '\n')
    log_f.flush()


parser = argparse.ArgumentParser(description='training')
parser.add_argument('-i', '--max-epoch', type=int, nargs='?', default=160,
                    help='max epoch')
parser.add_argument('-n', '--tag', type=str, nargs='?', default='default',
                    help='set log tag')
parser.add_argument('-b', '--single-batch-size', type=int, nargs='?', default=2,
                    help='set batch size')
parser.add_argument('-l', '--lr', type=float, nargs='?', default=0.0005,
                    help='set learning rate')
parser.add_argument('-al', '--alpha', type=float, nargs='?', default=1.0,
                    help='set alpha in cls_positive loss function')
parser.add_argument('-be', '--beta', type=float, nargs='?', default=10.0,
                    help='set beta in cls_negative loss function')
parser.add_argument('--output-path', type=str, nargs='?',
                    default='./predictions', help='results output dir')
parser.add_argument('-r', '--restore', type=bool, nargs='?', default=False,
                    help='set the flag to True if restore')
parser.add_argument('-v', '--vis', type=bool, nargs='?', default=False,
                    help='set the flag to True if dumping visualizations')
args = parser.parse_args()

dataset_dir = cfg.DATA_DIR
if cfg.USE_AUGED_DATA:
    AUG_DATA = False
    train_dir = os.path.join(cfg.DATA_DIR, 'training', cfg.AUG_DATA_FOLDER)
    val_dir = os.path.join(cfg.DATA_DIR, 'validation', cfg.AUG_DATA_FOLDER)
else:
    AUG_DATA = True
    train_dir = os.path.join(cfg.DATA_DIR, 'training50')
    val_dir = os.path.join(cfg.DATA_DIR, 'validation50')
log_dir = os.path.join('./log', args.tag)
save_model_dir = os.path.join('./save_model', args.tag)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(save_model_dir, exist_ok=True)

database_info_path = os.path.join(dataset_dir, "kitti_dbinfos_train.pkl")
sampler = DataBaseSampler(dataset_dir, database_info_path, global_rot_range=[0.5,0.5])
sampler.print_class_name()

def main(_):
    global log_f
    timestr = time.strftime("%b-%d_%H-%M-%S", time.localtime())
    log_f = open('log/train_{}.txt'.format(timestr), 'w')
    log_print(str(cfg))
    # TODO: split file support
    with tf.Graph().as_default():
        global save_model_dir
        start_epoch = 0
        global_counter = 0

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=cfg.GPU_MEMORY_FRACTION,
                                    visible_device_list=cfg.GPU_AVAILABLE,
                                    allow_growth=True)
        config = tf.ConfigProto(
            gpu_options=gpu_options,
            device_count={
                "GPU": cfg.GPU_USE_COUNT,
            },
            allow_soft_placement=True,
        )
        with tf.Session(config=config) as sess:
            model = RPN3D(
                cls=cfg.DETECT_OBJ,
                single_batch_size=args.single_batch_size,
                learning_rate=args.lr,
                max_gradient_norm=5.0,
                alpha=args.alpha,
                beta=args.beta,
                avail_gpus=cfg.GPU_AVAILABLE.split(',')
            )
            # param init/restore
            if args.restore and tf.train.get_checkpoint_state(save_model_dir):
                log_print("Reading model parameters from %s" % save_model_dir)
                model.saver.restore(sess, tf.train.latest_checkpoint(save_model_dir))
                start_epoch = model.epoch.eval() + 1
                global_counter = model.global_step.eval() + 1
            else:
                log_print("Created model with fresh parameters.")
                tf.global_variables_initializer().run()

            if cfg.FEATURE_NET_TYPE == 'FeatureNet_AE' and cfg.FeatureNet_AE_WPATH:
                ae_checkpoint_file = tf.train.latest_checkpoint(cfg.FeatureNet_AE_WPATH)
                log_print("Load Pretrained FeatureNet_AE weights %s" % ae_checkpoint_file)
                ae_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='ae_encoder')
                ae_saver = tf.train.Saver(var_list={v.op.name: v for v in ae_vars})
                ae_saver.restore(sess, ae_checkpoint_file)
            if cfg.FEATURE_NET_TYPE == 'FeatureNet_VAE' and cfg.FeatureNet_VAE_WPATH:
                vae_checkpoint_file = tf.train.latest_checkpoint(cfg.FeatureNet_VAE_WPATH)
                log_print("Load Pretrained FeatureNet_VAE weights %s" % vae_checkpoint_file)
                vae_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vae_encoder')
                vae_saver = tf.train.Saver(var_list={v.op.name: v for v in vae_vars})
                vae_saver.restore(sess, vae_checkpoint_file)

            # train and validate
            is_summary, is_summary_image, is_validate = False, False, False

            summary_interval = 5
            summary_val_interval = 20
            summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

            parameter_num = np.sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()])
            log_print('Parameter number: {}'.format(parameter_num))

            # training
            for epoch in range(start_epoch, args.max_epoch):
                counter = 0
                batch_time = time.time()
                for batch in iterate_data(train_dir, db_sampler=sampler, shuffle=True, aug=AUG_DATA, is_testset=False,
                                          batch_size=args.single_batch_size * cfg.GPU_USE_COUNT, multi_gpu_sum=cfg.GPU_USE_COUNT):
                    counter += 1
                    global_counter += 1

                    if counter % summary_interval == 0:
                        is_summary = True
                    else:
                        is_summary = False

                    start_time = time.time()
                    ret = model.train_step( sess, batch, train=True, summary = is_summary )
                    forward_time = time.time() - start_time
                    batch_time = time.time() - batch_time

                    log_print('train: {} @ epoch:{}/{} loss: {:.4f} reg_loss: {:.4f} cls_loss: {:.4f} cls_pos_loss: {:.4f} cls_neg_loss: {:.4f} forward time: {:.4f} batch time: {:.4f}'.
                                format(counter,epoch, args.max_epoch, ret[0], ret[1], ret[2], ret[3], ret[4], forward_time, batch_time), write=is_summary)

                    #print(counter, summary_interval, counter % summary_interval)
                    if counter % summary_interval == 0:
                        log_print("summary_interval now")
                        summary_writer.add_summary(ret[-1], global_counter)

                    #print(counter, summary_val_interval, counter % summary_val_interval)
                    if counter % summary_val_interval == 0:
                        log_print("summary_val_interval now")
                        # Random sample single batch data
                        batch = sample_test_data(val_dir, args.single_batch_size * cfg.GPU_USE_COUNT,
                                                 multi_gpu_sum=cfg.GPU_USE_COUNT)

                        ret = model.validate_step(sess, batch, summary=True)
                        summary_writer.add_summary(ret[-1], global_counter)
                        log_print('validation: loss: {:.4f} reg_loss: {:.4f} cls_loss: {:.4f} '.format(ret[0], ret[1], ret[2]))

                        with warnings.catch_warnings():
                            warnings.filterwarnings('error')
                            try:
                                ret = model.predict_step(sess, batch, summary=True)
                                summary_writer.add_summary(ret[-1], global_counter)
                            except:
                                log_print('prediction skipped due to error', 'red')

                    if check_if_should_pause(args.tag):
                        model.saver.save(sess, os.path.join(save_model_dir, timestr), global_step=model.global_step)
                        log_print('pause and save model @ {} steps:{}'.format(save_model_dir, model.global_step.eval()))
                        sys.exit(0)

                    batch_time = time.time()

                sess.run(model.epoch_add_op)

                model.saver.save(sess, os.path.join(save_model_dir, timestr), global_step=model.global_step)

                # dump test data every 10 epochs
                if ( epoch + 1 ) % 10 == 0:
                    # create output folder
                    os.makedirs(os.path.join(args.output_path, str(epoch)), exist_ok=True)
                    os.makedirs(os.path.join(args.output_path, str(epoch), 'data'), exist_ok=True)
                    if args.vis:
                        os.makedirs(os.path.join(args.output_path, str(epoch), 'vis'), exist_ok=True)

                    for batch in iterate_data(val_dir, shuffle=False, aug=False, is_testset=False,
                                              batch_size=args.single_batch_size * cfg.GPU_USE_COUNT, multi_gpu_sum=cfg.GPU_USE_COUNT):
                        if args.vis:
                            tags, results, front_images, bird_views, heatmaps = model.predict_step(sess, batch, summary=False, vis=True)
                        else:
                            tags, results = model.predict_step(sess, batch, summary=False, vis=False)

                        for tag, result in zip(tags, results):
                            of_path = os.path.join(args.output_path, str(epoch), 'data', tag + '.txt')
                            with open(of_path, 'w+') as f:
                                P, Tr, R = load_calib( os.path.join( cfg.CALIB_DIR, tag + '.txt' ) )
                                labels = box3d_to_label([result[:, 1:8]], [result[:, 0]], [result[:, -1]], coordinate='lidar',
                                                        P2=P, T_VELO_2_CAM=Tr, R_RECT_0=R)[0]
                                for line in labels:
                                    f.write(line)
                                log_print('write out {} objects to {}'.format(len(labels), tag))
                        # dump visualizations
                        if args.vis:
                            for tag, front_image, bird_view, heatmap in zip(tags, front_images, bird_views, heatmaps):
                                front_img_path = os.path.join( args.output_path, str(epoch),'vis', tag + '_front.jpg'  )
                                bird_view_path = os.path.join( args.output_path, str(epoch), 'vis', tag + '_bv.jpg'  )
                                heatmap_path = os.path.join( args.output_path, str(epoch), 'vis', tag + '_heatmap.jpg'  )
                                cv2.imwrite( front_img_path, front_image )
                                cv2.imwrite( bird_view_path, bird_view )
                                cv2.imwrite( heatmap_path, heatmap )

                    # execute evaluation code
                    cmd_1 = "./kitti_eval/launch_test.sh"
                    cmd_2 = os.path.join( args.output_path, str(epoch) )
                    cmd_3 = os.path.join( args.output_path, str(epoch), 'log' )
                    os.system( " ".join( [cmd_1, cmd_2, cmd_3] ) )

            log_print('train done. total epoch:{} iter:{}'.format(
                epoch, model.global_step.eval()))

            # finallly save model
            model.saver.save(sess, os.path.join(
                save_model_dir, 'checkpoint'), global_step=model.global_step)


if __name__ == '__main__':
    tf.app.run(main)
