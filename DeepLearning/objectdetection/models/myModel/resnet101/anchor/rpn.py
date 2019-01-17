import numpy as np

import tensorflow as tf
from keras.layers import Conv2D, BatchNormalization, Activation
from keras.layers import MaxPooling2D, Add, GlobalAveragePooling2D, Dense


def conv_bn(input_tensor,
            kernel_size,
            filters,
            padding='same',
            strides=(1, 1),
            name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        use_bias=False,
        kernel_initializer='he_normal',
        name=conv_name)(input_tensor)
    x = BatchNormalization(name=bn_name)(x)
    x = Activation('relu')(x)
    return x


def Scale(input_tensor):
    alpha = tf.Variable(tf.random_normal(1))
    bias = tf.Variable(0)
    return alpha * input_tensor + bias


def conv_scale(input_tensor,
               kernel_size,
               filters,
               padding='same',
               strides=(1, 1),
               name=None):
    x = Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        use_bias=False,
        kernel_initializer='he_normal',
        name=conv_name)(input_tensor)
    x = Scale(x)
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, strides, name=None):
    filters1, filters2, filters3 = filters
    with tf.name_scope(name + '_branch') as scope:
        branch = conv_scale(
            input_tensor, 1, filters1, padding='same', strides=1)
        branch = conv_scale(
            branch, 3, filters2, padding='same', strides=strides)
        branch = Conv2D(filters3, 1, strides=1, padding='same')(branch)

    shortcut = Conv2D(
        filters3, 1, strides=strides, padding='same')(input_tensor)
    return Add()([branch, shortcut])


def identity_block(input_tensor, kernel_size, filters, strides, name=None):
    filters1, filters2, filters3 = filters
    with tf.name_scope(name + '_branch') as scope:
        branch = Scale(input_tensor)
        branch = Activation('relu')(branch)
        branch = conv_scale(branch, 1, filters1, padding='same', strides=1)
        branch = conv_scale(
            branch, kernel_size, filters2, padding='same', strides=1)
        branch = Conv2D(filters3, 1, strides=1, padding='same')(branch)

    return Add()([branch, input_tensor])


def inference(images):
    """
    shape of images [1,3,600,600]
    """
    with tf.name_scope('conv1') as scope:
        x = conv_scale(
            images,
            kernel_size=7,
            filters=64,
            padding='same',
            strides=2,
            name='conv1')
        x = MaxPooling2D(pool_size=3, strides=2)(x)
    with tf.name_scope('res1') as scope:
        x = conv_block(x, 3, [64, 64, 256], strides=1, name=scope)
    for i in range(2, 4):
        with tf.name_scope('res' + str(i)) as scope:
            x = identity_block(
                x, kernel_size=3, [64, 64, 256], strides=1, name=scope)

    x = Scale(x)
    x = Activation('relu')(x)
    with tf.name_scope('res4') as scope:
        x = conv_block(x, 3, [128, 128, 512], strides=2, name=scope)

    for i in range(5, 8):
        with tf.name_scope('res' + str(i)) as scope:
            x = identity_block(
                x, kernel_size=3, [128, 128, 512], strides=1, name=scope)

    x = Scale(x)
    x = Activation('relu')(x)

    with tf.name_scope('res8') as scope:
        x = conv_block(x, 3, [256, 256, 1024], strides=2, name=scope)

    for i in range(9, 31):
        with tf.name_scope('res' + str(i)) as scope:
            x = identity_block(
                x, kernel_size=3, [256, 256, 1024], strides=1, name=scope)

    x = Scale(x)
    x = Activation('relu')(x)

    with tf.name_scope('rpn') as scope:
        x = Conv2D(
            512,
            kernel_size=3,
            strides=1,
            padding='same',
            activation='relu',
            name=scope)

    with tf.name_scope('rpn_bbox') as scope:
        rpn_bbox = Conv2D(
            36, kernel_size=1, strides=1, padding='same', name=scope)(x)

    with tf.name_scope('rpn_cls') as scope:
        rpn_cls=Conv2D(18,kernel_size=1,strides=1,padding='same')(x)
        rpn_cls=