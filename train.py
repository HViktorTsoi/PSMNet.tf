import tensorflow as tf

import config
import dataset
from PSMNet import PSMNet
import matplotlib.pyplot as plt


def train():
    with tf.Session() as sess:
        # 构建模型
        model = PSMNet(width=config.TRAIN_CROP_WIDTH, height=config.TRAIN_CROP_HEIGHT, channels=config.IMG_N_CHANNEL,
                       head_type=config.HEAD_STACKED_HOURGLASS, batch_size=config.TRAIN_BATCH_SIZE)
        model.build_net()

        saver = tf.train.Saver()

        # 载入数据集
        train_dataset, dataset_size = dataset.get_dataset(data_path='dataset/', batch_size=config.TRAIN_BATCH_SIZE,
                                                          epoch=config.TRAIN_EPOCH, num_threads=10, is_training=True)
        data_iterator = train_dataset.make_one_shot_iterator()
        next_batch = data_iterator.get_next()

        # 训练
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, config.TRAIN_EPOCH + 1):
            epoch_loss = 0
            step_ave_loss = 0
            for step in range(dataset_size // config.TRAIN_BATCH_SIZE):
                loss = model.train(
                    sess,
                    left_imgs=next_batch[0].eval(session=sess),
                    right_imgs=next_batch[1].eval(session=sess),
                    disp_gt=next_batch[2].eval(session=sess)
                )
                epoch_loss += loss
                step_ave_loss += loss
                if step % config.LOG_INTERVAL == 0 and step > 0:
                    print('EPOCH {:04d} - {:04d}/{:04d}  LOSS: {:.4f}'.format(
                        epoch, step, dataset_size // config.TRAIN_BATCH_SIZE,
                                     step_ave_loss / config.LOG_INTERVAL
                    ))
                    step_ave_loss = 0
            print('EPOCH LOSS: {}'.format(epoch_loss / (dataset_size // config.TRAIN_BATCH_SIZE)))
            # 保存模型
            saver.save(sess, save_path='./ckpt/scene_flow_hgls.ckpt', global_step=epoch)


def val():
    with tf.Session() as sess:
        # 构建模型
        model = PSMNet(width=config.TRAIN_CROP_WIDTH, height=config.TRAIN_CROP_HEIGHT, channels=config.IMG_N_CHANNEL,
                       head_type=config.HEAD_STACKED_HOURGLASS, batch_size=config.VAL_BATCH_SIZE)
        model.build_net()

        saver = tf.train.Saver()

        # 载入数据集
        val_dataset, dataset_size = dataset.get_dataset(data_path='dataset/', batch_size=config.VAL_BATCH_SIZE,
                                                        epoch=config.TRAIN_EPOCH, num_threads=10, is_training=True)
        data_iterator = val_dataset.make_one_shot_iterator()
        next_batch = data_iterator.get_next()

        # 验证
        saver.restore(sess, save_path='./ckpt/scene_flow_hgls.ckpt-1')
        for epoch in range(1, config.TRAIN_EPOCH + 1):
            for step in range(dataset_size // config.VAL_BATCH_SIZE):
                pred = sess.run(
                    [model.disparity_3],
                    feed_dict={
                        model.left_inputs: next_batch[0].eval(session=sess),
                        model.right_inputs: next_batch[1].eval(session=sess),
                        model.groundtruth: next_batch[2].eval(session=sess),
                        model.is_training: False
                    }
                )
                plt.imshow(pred[0][0])
                plt.show()


if __name__ == '__main__':
    # val()
    train()
