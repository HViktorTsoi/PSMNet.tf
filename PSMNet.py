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
        self.cost_volume = self.cost_volume_aggregation(self.ref_feature, self.target_feature, config.MAX_DISP)

        # # 实现3d cnn以及视差图估计
        # if self.head_type == config.HEAD_STACKED_HOURGLASS:
        #     self.disparity_1, self.disparity_2, self.disparity_3 = self.stacked_hourglass(self.cost_volume)
        # elif self.head_type == config.HEAD_BASIC:
        #     self.disparity = self.basic(self.cost_volume)
        # else:
        #     raise NotImplementedError('Head Type \'{}\' Not Supported!!!'.format(self.head_type))

    def cnn(self, inputs, weight_share=False):
        with tf.variable_scope('CNN_BASE'):
            outputs = inputs
            with tf.variable_scope('conv0'):
                # 第一层卷积+下采样 第二三层卷积
                for layer_id in range(3):
                    outputs = self._build_conv_block(
                        outputs, tf.layers.conv2d, filters=32, kernel_size=3,
                        strides=2 if layer_id == 0 else 1,
                        reuse=weight_share, layer_name='conv0_{}'.format(layer_id + 1)
                    )

            # 两层残差连接
            with tf.variable_scope('conv1'):
                # 三层res卷积
                for layer_id in range(3):
                    outputs = self._build_residual_block(
                        outputs, tf.layers.conv2d, filters=32, kernel_size=3,
                        reuse=weight_share, layer_name='res_conv1_{}'.format(layer_id + 1)
                    )

            with tf.variable_scope('conv2'):
                # 第一层两倍下采样 且包含投影 其余15层正常残差连接
                for layer_id in range(16):
                    outputs = self._build_residual_block(
                        outputs, tf.layers.conv2d, filters=64, kernel_size=3,
                        strides=2 if layer_id == 0 else 1, projection=(layer_id == 0),
                        reuse=weight_share, layer_name='res_conv2_{}'.format(layer_id + 1)
                    )

            # 两层空洞卷积
            with tf.variable_scope('conv3'):
                # 第一层包含投影 其余2层正常残差连接
                for layer_id in range(3):
                    outputs = self._build_residual_block(
                        outputs, tf.layers.conv2d, filters=128, kernel_size=3,
                        dilation_rate=2, projection=(layer_id == 0),
                        reuse=weight_share, layer_name='res_conv3_{}'.format(layer_id + 1)
                    )

            with tf.variable_scope('conv4'):
                # 三层res-block
                for layer_id in range(3):
                    outputs = self._build_residual_block(
                        outputs, tf.layers.conv2d, filters=128, kernel_size=3,
                        dilation_rate=4,
                        reuse=weight_share, layer_name='res_conv4_{}'.format(layer_id + 1)
                    )

            return outputs

    def spp(self, inputs, weight_share):
        with tf.variable_scope('SPP'):
            # spp的四个分支
            branches = [self._build_spp_branch(inputs, pool_size=pool_size,
                                               reuse=weight_share, layer_name='branch_{}'.format(branch_id + 1))
                        for branch_id, pool_size in enumerate([64, 32, 16, 8])]

            # 加上CNN base中的skip连接 conv2_16以及conv4_3(即inputs) 注意这里添加的是relu之前的连接
            branches.append(tf.get_default_graph().get_tensor_by_name('CNN_BASE/conv2/res_conv2_16/add:0'))
            branches.append(tf.get_default_graph().get_tensor_by_name('CNN_BASE/conv4/res_conv4_3/add:0'))

            # 拼接
            outputs = tf.concat(branches, axis=-1, name='spp_branch_concat')

            # 　特征融合
            fusion = self._build_conv_block(outputs, tf.layers.conv2d, filters=128, kernel_size=3,
                                            reuse=weight_share, layer_name='fusion_conv_3x3')
            fusion = self._build_conv_block(fusion, tf.layers.conv2d, filters=32, kernel_size=1,
                                            reuse=weight_share, layer_name='fusion_conv_1x1')
            print(outputs, fusion)
            return fusion

    def feature_extraction(self, inputs, weight_share=False):
        """
        特征提取层
        :param inputs: 输入
        :param weight_share: 是否共享权值
        :return: 特征
        """
        return self.spp(
            self.cnn(inputs, weight_share),
            weight_share
        )

    def cost_volume_aggregation(self, left_inputs, right_inputs, max_disp):
        """
        将左右两帧的特征聚合 生成cost-volume
        :param left_inputs: 左侧输入
        :param right_inputs: 右侧输入
        :param max_disp: 最大视差深度
        :return: cost-volume
        """
        cost_volume = []
        for d in range(max_disp // 4):
            if d > 0:
                # 左右两侧的视差cost 注意这里是在width维度计算的视差
                left_shift = left_inputs[:, :, d:, :]
                right_shift = left_inputs[:, :, :-d, :]

                # 补0填充 在width的维度起始补0
                left_shift = tf.pad(left_shift, paddings=[[0, 0], [0, 0], [d, 0], [0, 0]])
                right_shift = tf.pad(right_shift, paddings=[[0, 0], [0, 0], [d, 0], [0, 0]])

                # 在channel维度拼接
                cost_plate = tf.concat([left_shift, right_shift], axis=-1)
            else:
                # d为0时直接拼接未经过shift的原图
                cost_plate = tf.concat([left_inputs, right_inputs], axis=-1)
            cost_volume.append(cost_plate)

        # 将每个视差等级的cost图拼接成cost volume 注意要在第1个维度拼接 (第0个是batch)
        cost_volume = tf.stack(cost_volume, axis=1)

        return cost_volume

    def stacked_hourglass(self, inputs):
        return inputs, inputs, inputs

    def basic(self, inputs):
        return inputs

    def disparity_regression(self, inputs, pre):
        return inputs + pre

    def _build_conv_block(self, inputs, conv_function, filters, kernel_size, strides=1, dilation_rate=None,
                          layer_name='conv', apply_bn=True, apply_relu=True, reuse=False):
        # 构建卷积块
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
            outputs = conv_function(inputs, filters, kernel_size, strides, **conv_param)

            # bn
            if apply_bn:
                outputs = tf.layers.batch_normalization(
                    outputs, training=tf.get_default_graph().get_tensor_by_name('is_training:0'),
                    reuse=reuse, name='bn'
                )

            # 激活函数
            if apply_relu:
                outputs = tf.nn.relu(outputs)

            return outputs

    def _build_residual_block(self, inputs, conv_function, filters, kernel_size, strides=1, dilation_rate=None,
                              layer_name='conv', reuse=False, projection=False):
        # 构建残差连接块
        with tf.variable_scope(layer_name):
            inputs_shortcut = inputs
            # 构建res_block的前两个conv层
            outputs = self._build_conv_block(inputs, conv_function, filters, kernel_size, strides=strides,
                                             dilation_rate=dilation_rate, layer_name=layer_name + '_1', reuse=reuse)

            # 注意第二层没有relu 且strides=1(保证不进行下采样 下采样都由第一个conv完成)
            outputs = self._build_conv_block(outputs, conv_function, filters, kernel_size, strides=1,
                                             dilation_rate=dilation_rate, layer_name=layer_name + '_2',
                                             apply_relu=False, reuse=reuse)

            # 1x1投影 保证inputs_shortcut和outputs的channel一致
            if projection:
                inputs_shortcut = self._build_conv_block(inputs_shortcut, conv_function, filters, kernel_size=1,
                                                         strides=strides, layer_name='projection',
                                                         apply_relu=False, apply_bn=False, reuse=reuse)
            # 加残差连接
            outputs = tf.add(outputs, inputs_shortcut, name='add')
            outputs = tf.nn.relu(outputs, name='relu')
            return outputs

    def _build_spp_branch(self, inputs, pool_size, reuse, layer_name):
        # 构建spp的一个分支
        with tf.variable_scope(layer_name):
            # 原始尺寸
            origin_size = tf.shape(inputs)[1:3]

            # 平均池化
            outputs = tf.layers.average_pooling2d(inputs, pool_size, strides=1, name='avg_pool')

            # 卷积
            outputs = self._build_conv_block(outputs, tf.layers.conv2d, filters=32, kernel_size=3, reuse=reuse)

            # 上采样 恢复原始尺寸
            outputs = tf.image.resize_images(outputs, size=origin_size)

            return outputs


if __name__ == '__main__':
    print(config.TRAIN_CROP_WIDTH, config.TRAIN_CROP_HEIGHT, )
    psm_net = PSMNet(width=config.TRAIN_CROP_WIDTH, height=config.TRAIN_CROP_HEIGHT,
                     head_type=config.HEAD_STACKED_HOURGLASS, channels=config.IMG_N_CHANNEL, batch_size=18)
    psm_net.build_net()
    print(psm_net.ref_feature)
    print(psm_net.cost_volume)

    with tf.Session() as sess:
        writer = tf.summary.FileWriter("./log", sess.graph)
