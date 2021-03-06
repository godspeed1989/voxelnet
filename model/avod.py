import numpy as np
import tensorflow as tf

if __name__ != '__main__':
    from config import cfg
else:
    from easydict import EasyDict as edict
    cfg = edict()
    cfg.FEATURE_RATIO = 2

def avod_lite(inputs, training):
    # Encoder
    # 1 -> 1/2
    conv1 = Separable_Conv2D(inputs, Cout=32, k=3, s=(1, 1), dm=3,
                   pad='same', training=training, name='conv_e11')
    conv1 = Conv2D(conv1, Cout=32, k=3, s=(1, 1),
                   pad='same', training=training, name='conv_e12')
    pool1 = tf.layers.max_pooling2d(conv1, 2, strides=2, padding='valid')
    # 1/4
    conv2 = Conv2D(pool1, Cout=32, k=3, s=(1, 1),
                   pad='same', training=training, name='conv_e21')
    conv2 = Conv2D(conv2, Cout=32, k=3, s=(1, 1),
                   pad='same', training=training, name='conv_e22')
    pool2 = tf.layers.max_pooling2d(conv2, 2, strides=2, padding='valid')
    # 1/8
    conv3 = Conv2D(pool2, Cout=64, k=3, s=(1, 1),
                   pad='same', training=training, name='conv_e31')
    conv3 = Conv2D(conv3, Cout=64, k=3, s=(1, 1),
                   pad='same', training=training, name='conv_e32')
    conv3 = Conv2D(conv3, Cout=64, k=3, s=(1, 1),
                   pad='same', training=training, name='conv_e33')
    pool3 = tf.layers.max_pooling2d(conv3, 2, strides=2, padding='valid')
    # 1/8
    conv4 = Conv2D(pool3, Cout=128, k=3, s=(1, 1),
                   pad='same', training=training, name='conv_e41')
    conv4 = Conv2D(conv4, Cout=128, k=3, s=(1, 1),
                   pad='same', training=training, name='conv_e42')
    conv4 = Conv2D(conv4, Cout=128, k=3, s=(1, 1),
                   pad='same', training=training, name='conv_e43')

    # Decoder
    # 1/8 -> 1/4
    upconv3 = Deconv2D(conv4, Cout=128, k=3, s=(2, 2),
                       pad='same', training=training, name='upconv3')
    concat3 = tf.concat((conv3, upconv3), axis=3, name='concat3')
    pyramid_fusion3 = Conv2D(concat3, Cout=128, k=3, s=(1, 1),
                             pad='same', training=training, name='pyramid_fusion3')
    if cfg.FEATURE_RATIO == 4:
        return pyramid_fusion3

    # 1/4 -> 1/2
    upconv2 = Deconv2D(pyramid_fusion3, Cout=64, k=3, s=(2, 2),
                       pad='same', training=training, name='upconv2')
    concat2 = tf.concat((conv2, upconv2), axis=3, name='concat2')
    pyramid_fusion2 = Conv2D(concat2, Cout=64, k=3, s=(1, 1),
                             pad='same', training=training, name='pyramid_fusion2')
    if cfg.FEATURE_RATIO == 2:
        return pyramid_fusion2

    # 1/2 -> 1
    upconv1 = Deconv2D(pyramid_fusion2, Cout=32, k=3, s=(2, 2),
                       pad='same', training=training, name='upconv1')
    concat1 = tf.concat((conv1, upconv1), axis=3, name='concat1')
    pyramid_fusion1 = Conv2D(concat1, Cout=32, k=3, s=(1, 1),
                             pad='same', training=training, name='pyramid_fusion1')

    return pyramid_fusion1

def avod(inputs, training):
    # Encoder
    # 1 -> 1/2
    conv1 = Separable_Conv2D(inputs, Cout=32, k=3, s=(1, 1), dm=3,
                   pad='same', training=training, name='conv_e11')
    conv1 = Conv2D(conv1, Cout=32, k=3, s=(1, 1),
                   pad='same', training=training, name='conv_e12')
    pool1 = tf.layers.max_pooling2d(conv1, 2, strides=2, padding='valid')
    # 1/4
    conv2 = Conv2D(pool1, Cout=64, k=3, s=(1, 1),
                   pad='same', training=training, name='conv_e21')
    conv2 = Conv2D(conv2, Cout=64, k=3, s=(1, 1),
                   pad='same', training=training, name='conv_e22')
    pool2 = tf.layers.max_pooling2d(conv2, 2, strides=2, padding='valid')
    # 1/8
    conv3 = Conv2D(pool2, Cout=128, k=3, s=(1, 1),
                   pad='same', training=training, name='conv_e31')
    conv3 = Conv2D(conv3, Cout=128, k=3, s=(1, 1),
                   pad='same', training=training, name='conv_e32')
    conv3 = Conv2D(conv3, Cout=128, k=3, s=(1, 1),
                   pad='same', training=training, name='conv_e33')
    pool3 = tf.layers.max_pooling2d(conv3, 2, strides=2, padding='valid')
    # 1/8
    conv4 = Conv2D(pool3, Cout=256, k=3, s=(1, 1),
                   pad='same', training=training, name='conv_e41')
    conv4 = Conv2D(conv4, Cout=256, k=3, s=(1, 1),
                   pad='same', training=training, name='conv_e42')
    conv4 = Conv2D(conv4, Cout=256, k=3, s=(1, 1),
                   pad='same', training=training, name='conv_e43')

    # Decoder
    # 1/8 -> 1/4
    upconv3 = Deconv2D(conv4, Cout=256, k=3, s=(2, 2),
                       pad='same', training=training, name='upconv3')
    concat3 = tf.concat((conv3, upconv3), axis=3, name='concat3')
    pyramid_fusion3 = Conv2D(concat3, Cout=128, k=3, s=(1, 1),
                             pad='same', training=training, name='pyramid_fusion3')
    if cfg.FEATURE_RATIO == 4:
        return pyramid_fusion3

    # 1/4 -> 1/2
    upconv2 = Deconv2D(pyramid_fusion3, Cout=128, k=3, s=(2, 2),
                       pad='same', training=training, name='upconv2')
    concat2 = tf.concat((conv2, upconv2), axis=3, name='concat2')
    pyramid_fusion2 = Conv2D(concat2, Cout=64, k=3, s=(1, 1),
                             pad='same', training=training, name='pyramid_fusion2')
    if cfg.FEATURE_RATIO == 2:
        return pyramid_fusion2

    # 1/2 -> 1
    upconv1 = Deconv2D(pyramid_fusion2, Cout=64, k=3, s=(2, 2),
                       pad='same', training=training, name='upconv1')
    concat1 = tf.concat((conv1, upconv1), axis=3, name='concat1')
    pyramid_fusion1 = Conv2D(concat1, Cout=32, k=3, s=(1, 1),
                             pad='same', training=training, name='pyramid_fusion1')

    return pyramid_fusion1

def Conv2D_mob(inputs, Cout, k, s, pad, training, activation=True, bn=True, name='conv'):
    ret = conv_dw(inputs, name=name, out_channels=Cout, training=training)
    return ret

def avod_mob(inputs, training):
    # Encoder
    # 1 -> 1/2
    conv1 = Conv2D_mob(inputs, Cout=32, k=3, s=(1, 1),
                   pad='same', training=training, name='conv_e11')
    conv1 = Conv2D_mob(conv1, Cout=32, k=3, s=(1, 1),
                   pad='same', training=training, name='conv_e12')
    pool1 = tf.layers.max_pooling2d(conv1, 2, strides=2, padding='valid')
    # 1/4
    conv2 = Conv2D_mob(pool1, Cout=64, k=3, s=(1, 1),
                   pad='same', training=training, name='conv_e21')
    conv2 = Conv2D_mob(conv2, Cout=64, k=3, s=(1, 1),
                   pad='same', training=training, name='conv_e22')
    pool2 = tf.layers.max_pooling2d(conv2, 2, strides=2, padding='valid')
    # 1/8
    conv3 = Conv2D_mob(pool2, Cout=128, k=3, s=(1, 1),
                   pad='same', training=training, name='conv_e31')
    conv3 = Conv2D_mob(conv3, Cout=128, k=3, s=(1, 1),
                   pad='same', training=training, name='conv_e32')
    conv3 = Conv2D_mob(conv3, Cout=128, k=3, s=(1, 1),
                   pad='same', training=training, name='conv_e33')
    pool3 = tf.layers.max_pooling2d(conv3, 2, strides=2, padding='valid')
    # 1/8
    conv4 = Conv2D_mob(pool3, Cout=256, k=3, s=(1, 1),
                   pad='same', training=training, name='conv_e41')
    conv4 = Conv2D_mob(conv4, Cout=256, k=3, s=(1, 1),
                   pad='same', training=training, name='conv_e42')
    conv4 = Conv2D_mob(conv4, Cout=256, k=3, s=(1, 1),
                   pad='same', training=training, name='conv_e43')

    # Decoder
    # 1/8 -> 1/4
    upconv3 = Deconv2D(conv4, Cout=256, k=3, s=(2, 2),
                       pad='same', training=training, name='upconv3')
    concat3 = tf.concat((conv3, upconv3), axis=3, name='concat3')
    pyramid_fusion3 = Conv2D_mob(concat3, Cout=128, k=3, s=(1, 1),
                                 pad='same', training=training, name='pyramid_fusion3')
    if cfg.FEATURE_RATIO == 4:
        return pyramid_fusion3

    # 1/4 -> 1/2
    upconv2 = Deconv2D(pyramid_fusion3, Cout=128, k=3, s=(2, 2),
                       pad='same', training=training, name='upconv2')
    concat2 = tf.concat((conv2, upconv2), axis=3, name='concat2')
    pyramid_fusion2 = Conv2D_mob(concat2, Cout=64, k=3, s=(1, 1),
                                 pad='same', training=training, name='pyramid_fusion2')
    if cfg.FEATURE_RATIO == 2:
        return pyramid_fusion2

    # 1/2 -> 1
    upconv1 = Deconv2D(pyramid_fusion2, Cout=64, k=3, s=(2, 2),
                       pad='same', training=training, name='upconv1')
    concat1 = tf.concat((conv1, upconv1), axis=3, name='concat1')
    pyramid_fusion1 = Conv2D_mob(concat1, Cout=32, k=3, s=(1, 1),
                                 pad='same', training=training, name='pyramid_fusion1')

    return pyramid_fusion1

def Conv2D(inputs, Cout, k, s, pad, training, activation=True, bn=True, name='conv'):
    with tf.variable_scope(name) as scope:
        temp_conv = tf.layers.conv2d(
            inputs, Cout, k, strides=s, padding=pad, use_bias=False, reuse=tf.AUTO_REUSE, name=scope)
        if bn:
            temp_conv = tf.layers.batch_normalization(
                temp_conv, axis=-1, fused=True, training=training, reuse=tf.AUTO_REUSE, name=scope)
        if activation:
            return tf.nn.relu(temp_conv)
        else:
            return temp_conv

def Separable_Conv2D(inputs, Cout, k, s, pad, dm, training, activation=True, bn=True, name='conv'):
    with tf.variable_scope(name) as scope:
        temp_conv = tf.layers.separable_conv2d(
            inputs, Cout, k, strides=s, padding=pad, use_bias=False, depth_multiplier=dm, reuse=tf.AUTO_REUSE, name=scope)
        if bn:
            temp_conv = tf.layers.batch_normalization(
                temp_conv, axis=-1, fused=True, training=training, reuse=tf.AUTO_REUSE, name=scope)
        if activation:
            return tf.nn.relu(temp_conv)
        else:
            return temp_conv

def Deconv2D(inputs, Cout, k, s, pad, training, bn=True, name='deconv'):
    with tf.variable_scope(name) as scope:
        temp_conv = tf.layers.conv2d_transpose(
            inputs, Cout, k, strides=s, padding=pad, use_bias=False, reuse=tf.AUTO_REUSE, name=scope)
        if bn:
            temp_conv = tf.layers.batch_normalization(
                temp_conv, axis=-1, fused=True, training=training, reuse=tf.AUTO_REUSE, name=scope)
        return tf.nn.relu(temp_conv)

if __name__ == '__main__':
    inputs = tf.placeholder(tf.float32, [None, 360, 1200, 32])
    training = tf.placeholder(tf.bool)
    ret = avod(inputs, training)
    parameter_num1 = np.sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()])
    print('Parameter number: {}'.format(parameter_num1))
    print(ret.shape)
