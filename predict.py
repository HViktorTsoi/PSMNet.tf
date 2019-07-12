import os

import tensorflow as tf

import config
from PSMNet import PSMNet
from dataloader.data_loader import DataLoaderSceneFlow, DataLoaderKITTI_SUBMISSION, DataLoaderKITTI
from train import val


def sceneflow_predict(ckpt_path, vis=True, save_fig=True):
    """
    scene flow测试
    :param ckpt_path:
    :param vis:
    :return:
    """
    with tf.Session() as sess:
        # 构建模型
        model = PSMNet(width=config.TRAIN_CROP_WIDTH, height=config.TRAIN_CROP_HEIGHT, channels=config.IMG_N_CHANNEL,
                       head_type=config.HEAD_STACKED_HOURGLASS, batch_size=config.VAL_BATCH_SIZE)
        model.build_net()

        saver = tf.train.Saver()
        saver.restore(sess, save_path=ckpt_path)

        test_loader = DataLoaderSceneFlow(batch_size=config.TRAIN_BATCH_SIZE, max_disp=config.MAX_DISP)

        val(sess, model, data_loader=test_loader, vis=vis, save_fig=save_fig)


def kitti_predict(ckpt_path, vis=True, save_fig=True):
    """
    scene flow测试
    :param ckpt_path:
    :param vis:
    :return:
    """
    with tf.Session() as sess:
        # 构建模型
        # model = PSMNet(width=config.KITTI2015_SIZE[1], height=config.KITTI2015_SIZE[0], channels=config.IMG_N_CHANNEL,
        #                head_type=config.HEAD_STACKED_HOURGLASS, batch_size=config.VAL_BATCH_SIZE)
        model = PSMNet(width=config.TRAIN_CROP_WIDTH, height=config.TRAIN_CROP_HEIGHT, channels=config.IMG_N_CHANNEL,
                       head_type=config.HEAD_STACKED_HOURGLASS, batch_size=config.VAL_BATCH_SIZE)
        model.build_net()

        saver = tf.train.Saver()
        saver.restore(sess, save_path=ckpt_path)

        # test_loader = DataLoaderKITTI_SUBMISSION()
        test_loader = DataLoaderKITTI(batch_size=config.TRAIN_BATCH_SIZE, max_disp=config.MAX_DISP)

        val(sess, model, data_loader=test_loader, vis=vis, save_fig=save_fig)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # sceneflow_predict(ckpt_path='./ckpt/scene_flow_hgls.ckpt-6', vis=False, save_fig=True)
    kitti_predict(ckpt_path='./ckpt/KITTI_hgls.ckpt-27', vis=True, save_fig=True)
