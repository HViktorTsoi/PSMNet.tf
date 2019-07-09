import tensorflow as tf
import tensorflow.contrib as tfc

import config


class PSMNet:

    def __init__(self, width, height, channels, batch_size, head_type, is_training=True):
        self.img_width = width
        self.img_height = height
        self.channels = channels
        self.batch_size = batch_size
        self.head_type = head_type
        # 模型输入 左右两张图
        self.left_inputs = tf.placeholder(tf.float32,
                                          (None, self.img_height, self.img_width, self.channels), name='left_inputs')
        self.right_inputs = tf.placeholder(tf.float32,
                                           (None, self.img_height, self.img_width, self.channels), name='right_inputs')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.estimation = None

    def build_net(self):
        # 提特征 注意建图时右分支和左分支共享权重
        self.ref_feature = self.feature_extraction(self.left_inputs)
        self.target_feature = self.feature_extraction(self.right_inputs, weight_share=True)

        # 计算cost volume
        self.cost_volume = self.cost_volume_aggregation(self.ref_feature, self.target_feature)

        # 实现3d cnn以及视差图估计
        if self.head_type == config.HEAD_STACKED_HOURGLASS:
            self.disparity_1, self.disparity_2, self.disparity_3 = self.stacked_hourglass(self.cost_volume)
        elif self.head_type == config.HEAD_BASIC:
            self.disparity = self.basic(self.cost_volume)
        else:
            raise NotImplementedError('Head Type \'{}\' Not Supported!!!'.format(self.head_type))

    def cnn(self, inputs, weight_share=False):
        with tf.variable_scope('CNN_BASE'):
            with tf.variable_scope('conv0'):
                # 第一层卷积+下采样
                inputs = self._build_conv_block(
                    inputs, tf.layers.conv2d,
                    filters=32, kernel_size=3, strides=2, reuse=weight_share, layer_name='conv0_1'
                )
                # 第二三层卷积
                for layer_id in range(1, 3):
                    inputs = self._build_conv_block(
                        inputs, tf.layers.conv2d, reuse=weight_share,
                        filters=32, kernel_size=3, layer_name='conv0_{}'.format(layer_id + 1)
                    )
            return inputs

    def spp(self, inputs):
        return inputs

    def feature_extraction(self, inputs, weight_share=False):
        return self.spp(self.cnn(inputs, weight_share))

    def cost_volume_aggregation(self, left_inputs, right_inputs):
        return tf.tile(tf.stack([left_inputs, right_inputs], axis=1), multiples=[1, 32, 1, 1, 1])

    def stacked_hourglass(self, inputs):
        return inputs, inputs, inputs

    def basic(self, inputs):
        return inputs

    def disparity_regression(self, inputs, pre):
        return inputs + pre

    def _build_conv_block(self, inputs, conv_function, filters, kernel_size, strides=1, dilation_rate=None,
                          layer_name='conv', apply_bn=True, apply_relu=True, reuse=False):
        conv_param = {
            'padding': 'same',
            'kernel_initializer': tfc.layers.xavier_initializer(),
            'kernel_regularizer': tfc.layers.l2_regularizer(config.L2_REG),
            'bias_regularizer': tfc.layers.l2_regularizer(config.L2_REG),
            'reuse': reuse
        }
        if dilation_rate:
            conv_param['dilation_rate'] = dilation_rate
        # 构建卷积块
        with tf.variable_scope(layer_name):
            # 卷积
            inputs = conv_function(inputs, filters, kernel_size, strides, **conv_param)

            # bn
            if apply_bn:
                inputs = tf.layers.batch_normalization(
                    inputs, training=tf.get_default_graph().get_tensor_by_name('is_training:0'),
                    reuse=reuse, name='bn'
                )

            # 激活函数
            if apply_relu:
                inputs = tf.nn.relu(inputs)

            return inputs

    def _build_residual_block(self):
        pass


if __name__ == '__main__':
    psm_net = PSMNet(width=config.TRAIN_CROP_WIDTH, height=config.TRAIN_CROP_HEIGHT,
                     head_type=config.HEAD_STACKED_HOURGLASS, channels=config.IMG_N_CHANNEL, batch_size=18)
    psm_net.build_net()
    print(psm_net.ref_feature)
