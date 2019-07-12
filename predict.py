import cv2
import os

import tensorflow as tf

import config
import utils
from PSMNet import PSMNet
from dataloader.data_loader import DataLoaderSceneFlow, DataLoaderKITTI_SUBMISSION, DataLoaderKITTI
from train import val
import matplotlib.pyplot as plt


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
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    with tf.Session(config=gpu_config) as sess:
        # 构建模型
        model = PSMNet(width=config.KITTI2015_SIZE[1], height=config.KITTI2015_SIZE[0], channels=config.IMG_N_CHANNEL,
                       head_type=config.HEAD_STACKED_HOURGLASS, batch_size=config.VAL_BATCH_SIZE)
        # model = PSMNet(width=config.TRAIN_CROP_WIDTH, height=config.TRAIN_CROP_HEIGHT, channels=config.IMG_N_CHANNEL,
        #                head_type=config.HEAD_STACKED_HOURGLASS, batch_size=config.VAL_BATCH_SIZE)
        model.build_net()

        saver = tf.train.Saver()
        saver.restore(sess, save_path=ckpt_path)

        test_loader = DataLoaderKITTI_SUBMISSION()
        # test_loader = DataLoaderKITTI(batch_size=config.TRAIN_BATCH_SIZE, max_disp=config.MAX_DISP)

        # 验证
        for img_id, (imgL_crop, imgR_crop, groundtruth) in enumerate(test_loader.generator(is_training=False)):
            prediction = model.predict(
                sess,
                left_imgs=imgL_crop,
                right_imgs=imgR_crop,
            )

            # 可视化
            if save_fig:
                cv2.imwrite('./vis/{}'.format(test_loader.test_left_img[img_id].split('/')[-1]),
                            (prediction[0] * 256 * 1.17).astype('uint16'))

            if vis:
                plt.gcf().set_size_inches(10, 4)
                plt.subplot(1, 2, 1)
                plt.imshow(imgL_crop[0] * 0.25 + 0.4)
                plt.subplot(1, 2, 2)
                plt.imshow(prediction[0], cmap=plt.get_cmap('rainbow'))
                plt.tight_layout()
                plt.show()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # sceneflow_predict(ckpt_path='./ckpt/scene_flow_hgls.ckpt-6', vis=False, save_fig=True)
    kitti_predict(ckpt_path='./ckpt/KITTI_hgls.ckpt-27', vis=True, save_fig=True)
