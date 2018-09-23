
import numpy as np
import tensorflow as tf

if __name__ != '__main__':
    from config import cfg

def BasicConv(x, name, out_channels, kernel_size, training, stride=1,
                padding=(0, 0), dilation=1, bias=False, bn=True, relu=True):
    temp_p = np.array(padding)
    temp_p = np.lib.pad(temp_p, (1, 1), 'constant', constant_values=(0, 0))
    paddings = (np.array(temp_p)).repeat(2).reshape(4, 2)
    with tf.variable_scope(name):
        x = tf.pad(x, paddings, "CONSTANT")
        x = tf.layers.conv2d(x, filters=out_channels, kernel_size=kernel_size, strides=stride,
                            padding='VALID', dilation_rate=dilation, use_bias=bias)
        if bn:
            x = tf.layers.batch_normalization(x, fused=True, training=training)
        if relu:
            x = tf.nn.relu(x)
        return x

def create_variable(name, shape):
    initializer = tf.truncated_normal_initializer(stddev=1e-3)
    var = tf.get_variable(name, shape, initializer=initializer)
    return var

def BasicSepConv(x, name, kernel_size, training, stride=1,
                padding=0, dilation=1, bias=False, bn=True, relu=True):
    temp_p = np.array((padding, padding))
    temp_p = np.lib.pad(temp_p, (1, 1), 'constant', constant_values=(0, 0))
    paddings = (np.array(temp_p)).repeat(2).reshape(4, 2)
    with tf.variable_scope(name):
        x = tf.pad(x, paddings, "CONSTANT")
        num_in_channels = x.get_shape()[-1].value
        kernel = create_variable('weights', [kernel_size, kernel_size, num_in_channels, 1])

        x = tf.nn.depthwise_conv2d_native(x, filter=kernel, strides=[1, stride, stride, 1],
                                        padding='VALID', dilations=[1, dilation, dilation, 1])
        if bn:
            x = tf.layers.batch_normalization(x, fused=True, training=training)
        if relu:
            x = tf.nn.relu(x)
        return x

def BasicRFB(inputs, training):
    # in_channel >= 16
    in_planes = inputs.get_shape()[-1].value
    out_planes = in_planes
    inter_planes = in_planes // 4

    branch0 = BasicConv(inputs, 'branch0_0', inter_planes, kernel_size=1, training=training, stride=1)
    branch0 = BasicSepConv(branch0, 'branch0_1', kernel_size=3, training=training, stride=1, padding=1, dilation=1, relu=False)
    #print(branch0.shape)

    branch1 = BasicConv(inputs, 'branch1_0', inter_planes, kernel_size=1, training=training, stride=1)
    branch1 = BasicConv(branch1, 'branch1_1', inter_planes, kernel_size=(3,1), training=training, stride=1, padding=(1,0))
    branch1 = BasicSepConv(branch1, 'branch1_2', kernel_size=3, training=training, stride=1, padding=1, dilation=3, relu=False)
    #print(branch1.shape)

    branch2 = BasicConv(inputs, 'branch2_0', inter_planes, kernel_size=1, training=training, stride=1)
    branch2 = BasicConv(branch2, 'branch2_1', inter_planes, kernel_size=(1,3), training=training, stride=1, padding=(0,1))
    branch2 = BasicSepConv(branch2, 'branch2_2', kernel_size=3, training=training, stride=1, padding=1, dilation=3, relu=False)
    #print(branch2.shape)

    branch3 = BasicConv(inputs, 'branch3_0', inter_planes//2, kernel_size=1, training=training, stride=1)
    branch3 = BasicConv(branch3, 'branch3_1', (inter_planes//4)*3, kernel_size=(1,3), training=training, stride=1, padding=(0,1))
    branch3 = BasicConv(branch3, 'branch3_2', inter_planes, kernel_size=(3,1), training=training, stride=1, padding=(1,0))
    branch3 = BasicSepConv(branch3, 'branch3_3', kernel_size=3, training=training, stride=1, padding=1, dilation=5, relu=False)
    #print(branch3.shape)

    out = tf.concat([branch0, branch1, branch2, branch3], axis=-1)
    out = BasicConv(out, 'ConvLinear', out_planes, kernel_size=1, training=training, stride=1, relu=False)
    out = out + inputs
    out = tf.nn.relu(out)
    #print(out.shape)
    return out

def conv_dw(inputs, name, out_channels, training):
    with tf.variable_scope(name):
        x = BasicSepConv(inputs, 'conv0', kernel_size=3, training=training, stride=1, padding=1,
                         bias=False, bn=True, relu=True)
        x = BasicConv(x, 'conv1', out_channels, kernel_size=1, training=training, stride=1, padding=(0,0),
                      bias=False, bn=True, relu=True)
        return x

def RFBNet(inputs, training):
    if __name__ != '__main__':
        assert cfg.FEATURE_RATIO == 2
    x = conv_dw(inputs, 'conv_dw0', 32, training)
    x = conv_dw(x, 'conv_dw1', 64, training)
    x = BasicConv(x, 'conv_bn', 64, kernel_size=3, training=training, stride=2, padding=(1,1), bn=True, relu=True)
    x = conv_dw(x, 'conv_dw2', 128, training)
    x = conv_dw(x, 'conv_dw3', 256, training)
    s = BasicRFB(x, training)

    return s

if __name__ == '__main__':
    img = tf.placeholder(tf.float32, [11, 300, 400, 16])
    training = tf.placeholder(tf.bool)

    feat = RFBNet(img, training)
    print(feat.get_shape().as_list())

    parameter_num = np.sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()])
    print('parameters count {}'.format(parameter_num))