import numpy as np

import tensorflow as tf
from keras.layers import Conv2D, BatchNormalization, Activation
from keras.layers import MaxPooling2D, AveragePooling2D, Concatenate, Add, Dropout, Flatten, Dense, GlobalAveragePooling2D


def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
        relu_name = name + '_relu'
    else:
        bn_name = None
        conv_name = None
        relu_name = None

    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = BatchNormalization(name=bn_name)(x)
    x = Activation('relu', name=relu_name)(x)
    return x


def inception_resnet_a(input_tensor, name):
    with tf.name_scope('branch_0'):
        branch_0 = conv2d_bn(
            input_tensor,
            32,
            1,
            1,
            padding='valid',
            strides=1,
            name=(name + '_1x1'))

    with tf.name_scope('branch_1'):
        branch_1 = conv2d_bn(
            input_tensor,
            32,
            1,
            1,
            padding='valid',
            strides=1,
            name=(name + '_3x3_reduce'))
        branch_1 = conv2d_bn(
            branch_1,
            32,
            3,
            3,
            padding='same',
            strides=1,
            name=(name + '_3x3'))

    with tf.name_scope('branch_2'):
        branch_2 = conv2d_bn(
            input_tensor,
            32,
            1,
            1,
            padding='valid',
            strides=1,
            name=(name + '_3x3_2_reduce'))
        branch_2 = conv2d_bn(
            branch_2,
            48,
            3,
            3,
            padding='same',
            strides=1,
            name=(name + '_3x3_2'))
        branch_2 = conv2d_bn(
            branch_2,
            64,
            3,
            3,
            padding='same',
            strides=1,
            name=(name + '_3x3_3'))

    with tf.name_scope('concat'):
        branch_concat = Concatenate(name=(name + '_concat'))(
            [branch_0, branch_1, branch_2])
        branch_up = Conv2D(
            320, 1, strides=1, padding='same',
            name=(name + '_up'))(branch_concat)

    output = Add(name=(name + '_add'))([input_tensor, branch_up])
    output = Activation('relu', name=(name + '_add_relu'))(output)
    return output


def reduction_a(input_tensor, name):
    with tf.name_scope('branch_0'):
        branch_0 = conv2d_bn(
            input_tensor,
            384,
            3,
            3,
            padding='valid',
            strides=2,
            name=(name + '_3x3'))

    with tf.name_scope('branch_1'):
        branch_1 = conv2d_bn(
            input_tensor,
            256,
            1,
            1,
            padding='valid',
            strides=1,
            name=(name + '_3x3_2_reduce'))
        branch_1 = conv2d_bn(
            branch_1,
            256,
            3,
            3,
            padding='same',
            strides=1,
            name=(name + '_3x3_2'))
        branch_1 = conv2d_bn(
            branch_1,
            384,
            3,
            3,
            padding='valid',
            strides=2,
            name=(name + '_3x3_3'))

    with tf.name_scope('branch_2'):
        branch_2 = MaxPooling2D(
            3, strides=2, name=(name + '_pool'))(input_tensor)

    return Concatenate(name=(name + '_concat'))([branch_0, branch_1, branch_2])


def inception_resnet_b(input_tensor, name):
    with tf.name_scope('branch_0'):
        branch_0 = conv2d_bn(
            input_tensor,
            192,
            1,
            1,
            padding='valid',
            strides=1,
            name=(name + '_1x1'))

    with tf.name_scope('branch_1'):
        branch_1 = conv2d_bn(
            input_tensor,
            128,
            1,
            1,
            padding='valid',
            strides=1,
            name=(name + '_1x7_reduce'))
        branch_1 = conv2d_bn(
            branch_1,
            160,
            1,
            7,
            padding='same',
            strides=1,
            name=(name + '_1x7'))
        branch_1 = conv2d_bn(
            branch_1,
            192,
            7,
            1,
            padding='same',
            strides=1,
            name=(name + '_7x1'))

    with tf.name_scope('concat'):
        branch_concat = Concatenate(name=(name + '_concat'))(
            [branch_0, branch_1])
        branch_up = Conv2D(
            1088, 1, strides=1, padding='same',
            name=(name + '_up'))(branch_concat)

    output = Add(name=(name + '_add'))([input_tensor, branch_up])
    output = Activation('relu', name=(name + '_add_relu'))(output)
    return output


def reduction_b(input_tensor, name):
    with tf.name_scope('branch_0'):
        branch_0 = conv2d_bn(
            input_tensor,
            256,
            1,
            1,
            padding='valid',
            strides=1,
            name=(name + '_3x3_reduce'))
        branch_0 = conv2d_bn(
            branch_0,
            384,
            3,
            3,
            padding='valid',
            strides=2,
            name=(name + '_3x3'))

    with tf.name_scope('branch_1'):
        branch_1 = conv2d_bn(
            input_tensor,
            256,
            1,
            1,
            padding='valid',
            strides=1,
            name=(name + '_3x3_2_reduce'))
        branch_1 = conv2d_bn(
            branch_1,
            288,
            3,
            3,
            padding='valid',
            strides=2,
            name=(name + '_3x3_2'))

    with tf.name_scope('branch_2'):
        branch_2 = conv2d_bn(
            input_tensor,
            256,
            1,
            1,
            padding='valid',
            strides=1,
            name=(name + '_3x3_3_reduce'))
        branch_2 = conv2d_bn(
            branch_2,
            288,
            3,
            3,
            padding='same',
            strides=1,
            name=(name + '_3x3_3'))
        branch_2 = conv2d_bn(
            branch_2,
            320,
            3,
            3,
            padding='valid',
            strides=2,
            name=(name + '_3x3_4'))

    with tf.name_scope('branch_3'):
        branch_3 = MaxPooling2D(
            3, strides=2, name=(name + '_pool'))(input_tensor)

    return Concatenate(name=(name + '_concat'))(
        [branch_0, branch_1, branch_2, branch_3])


def inception_resnet_c(input_tensor, name):
    with tf.name_scope('branch_0'):
        branch_0 = conv2d_bn(
            input_tensor,
            192,
            1,
            1,
            padding='same',
            strides=1,
            name=(name + '_1x1'))

    with tf.name_scope('branch_1'):
        branch_1 = conv2d_bn(
            input_tensor,
            192,
            1,
            1,
            padding='same',
            strides=1,
            name=(name + '_1x3_reduce'))
        branch_1 = conv2d_bn(
            branch_1,
            224,
            1,
            3,
            padding='same',
            strides=1,
            name=(name + '_1x3'))
        branch_1 = conv2d_bn(
            branch_1,
            256,
            3,
            1,
            padding='same',
            strides=1,
            name=(name + '_3x1'))

    with tf.name_scope('concat'):
        branch_concat = Concatenate(name=(name + '_concat'))(
            [branch_0, branch_1])
        branch_up = Conv2D(
            2080, 1, strides=1, padding='same',
            name=(name + '_up'))(branch_concat)

    output = Add(name=(name + '_add'))([input_tensor, branch_up])
    output = Activation('relu', name=(name + '_add_relu'))(output)
    return output


def reduction_c(input_tensor, name):
    with tf.name_scope('branch_0'):
        branch_0 = conv2d_bn(
            input_tensor,
            192,
            1,
            1,
            padding='same',
            strides=1,
            name=(name + '_1x1'))

    with tf.name_scope('branch_1'):
        branch_1 = conv2d_bn(
            input_tensor,
            192,
            1,
            1,
            padding='same',
            strides=1,
            name=(name + '_1x3_reduce'))
        branch_1 = conv2d_bn(
            branch_1,
            224,
            1,
            3,
            padding='same',
            strides=1,
            name=(name + '_1x3'))
        branch_1 = conv2d_bn(
            branch_1,
            256,
            3,
            1,
            padding='same',
            strides=1,
            name=(name + '_3x1'))

    with tf.name_scope('concat'):
        branch_concat = Concatenate(name=(name + '_concat'))(
            [branch_0, branch_1])
        branch_up = Conv2D(
            2080, 1, strides=1, padding='same',
            name=(name + '_up'))(branch_concat)

    return Add(name=(name + '_add'))([input_tensor, branch_up])