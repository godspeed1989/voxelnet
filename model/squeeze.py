import numpy as np
import tensorflow as tf

def fire_module(inputs, squeeze_depth, expand_depth, scope):
    with tf.variable_scope(scope):
        net = _squeeze(inputs, squeeze_depth)
        net = _expand(net, expand_depth)
    return net

def _squeeze(inputs, num_outputs):
    return tf.layers.conv2d(inputs, num_outputs, [1, 1], strides=1, name='squeeze')

def _expand(inputs, num_outputs):
    with tf.variable_scope('expand'):
        e1x1 = tf.layers.conv2d(inputs, num_outputs, [1, 1], strides=1, name='1x1')
        e3x3 = tf.layers.conv2d(inputs, num_outputs, [3, 3], strides=1, padding='same', name='3x3')
    return tf.concat([e1x1, e3x3], axis=-1)

def res_fire_block(inputs, squeeze_depth, expand_depth, training, scope):
    C = inputs.get_shape()[-1]
    with tf.variable_scope(scope):
        net = tf.layers.batch_normalization(inputs, axis=-1, fused=True,
                                            training=training, reuse=tf.AUTO_REUSE, name='bn1')
        net = tf.nn.relu(net)
        net = fire_module(net, squeeze_depth, expand_depth, 'fire1')
        net = tf.layers.batch_normalization(net, axis=-1, fused=True,
                                            training=training, reuse=tf.AUTO_REUSE, name='bn2')
        net = tf.nn.relu(net)
        net = fire_module(net, squeeze_depth, C//2, 'fire2')
        net = tf.add(inputs, net)
    return net

def res_squeeze_net(inputs, training):
    net = tf.layers.conv2d(inputs, 128, [2, 2], strides=2, padding='valid', name='bottleneck')
    #
    net = tf.layers.conv2d(net, 128, [1, 1], strides=1, padding='valid', name='conv1')
    net = res_fire_block(net, 16, 64, training, 'block11')
    net = res_fire_block(net, 16, 64, training, 'block12')
    net = res_fire_block(net, 32, 128, training, 'block13')
    net1 = net
    #
    net = tf.layers.conv2d(net, 256, [1, 1], strides=1, padding='valid', name='conv2')
    net = res_fire_block(net, 32, 128, training, 'block21')
    net = res_fire_block(net, 32, 128, training, 'block22')
    net = res_fire_block(net, 48, 192, training, 'block23')
    net2 = net
    #
    net = tf.layers.conv2d(net, 384, [1, 1], strides=1, padding='valid', name='conv3')
    net = res_fire_block(net, 48, 192, training, 'block31')
    net = res_fire_block(net, 48, 192, training, 'block32')
    net = res_fire_block(net, 64, 256, training, 'block33')
    net3 = net

    net = tf.concat([net1, net2, net3], -1)
    return net

def res_block(inputs, bottleneck, training, scope):
    C = inputs.get_shape()[-1]
    with tf.variable_scope(scope):
        net = tf.layers.batch_normalization(inputs, axis=-1, fused=True,
                                            training=training, reuse=tf.AUTO_REUSE, name='bn1')
        net = tf.nn.relu(net)
        net = tf.layers.conv2d(net, bottleneck, [1, 1], strides=1, name='conv1x1')
        net = tf.layers.batch_normalization(net, axis=-1, fused=True,
                                            training=training, reuse=tf.AUTO_REUSE, name='bn2')
        net = tf.nn.relu(net)
        net = tf.layers.conv2d(net, C, [3, 3], strides=1, padding='same', name='conv3x3')
        net = tf.add(inputs, net)
    return net

def res_net(inputs, training):
    #
    net = tf.layers.conv2d(inputs, 32, [1, 1], strides=1, padding='valid', name='conv1')
    net = res_block(net, 32, training, 'block11')
    net = res_block(net, 64, training, 'block12')
    net = res_block(net, 128, training, 'block13')
    net = tf.layers.max_pooling2d(net, 2, strides=2, padding='valid')
    net1 = net
    #
    net = tf.layers.conv2d(net, 64, [1, 1], strides=1, padding='valid', name='conv2')
    net = res_block(net, 64, training, 'block21')
    net = res_block(net, 128, training, 'block22')
    net = res_block(net, 256, training, 'block23')
    net2 = net
    #
    net = tf.layers.conv2d(net, 128, [1, 1], strides=1, padding='valid', name='conv3')
    net = res_block(net, 128, training, 'block31')
    net = res_block(net, 256, training, 'block32')
    net = res_block(net, 384, training, 'block33')
    net3 = net

    net = tf.concat([net1, net2, net3], -1)
    return net


if __name__ == '__main__':
    inputs = tf.placeholder(tf.float32, [None, 400, 500, 32])
    training = tf.placeholder(tf.bool, name='phase')

    out = fire_module(inputs, 12, 34, 'fire1') # 68
    parameter_num1 = np.sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()])
    print('{} Parameter number: {}'.format(out.shape, parameter_num1))

    with tf.variable_scope('res_squeeze_net'):
        net = res_squeeze_net(inputs, training)
        parameter_num2 = np.sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()])
        print('{} Parameter number: {}'.format(net.shape, parameter_num2 - parameter_num1))

    with tf.variable_scope('res_net'):
        net = res_net(inputs, training)
        parameter_num3 = np.sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()])
        print('{} Parameter number: {}'.format(net.shape, parameter_num3 - parameter_num2))
