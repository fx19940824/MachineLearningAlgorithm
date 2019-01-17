import numpy as np

import tensorflow as tf
from keras.layers import Conv2D, BatchNormalization, Activation
from keras.layers import MaxPooling2D, AveragePooling2D, Concatenate, Dropout, Flatten, Dense, GlobalAveragePooling2D


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


def inception_a(input_tensor, name=None):
    with tf.variable_scope(name):
        with tf.name_scope('branch_0'):
            branch_0 = conv2d_bn(
                input_tensor,
                96,
                1,
                1,
                padding='same',
                strides=1,
                name=(name + '_1x1'))
        with tf.name_scope('branch_1'):
            branch_1 = conv2d_bn(
                input_tensor,
                64,
                1,
                1,
                padding='same',
                strides=1,
                name=(name + '_3x3_reduce'))
            branch_1 = conv2d_bn(
                branch_1,
                96,
                3,
                3,
                padding='same',
                strides=1,
                name=(name + '_3x3'))
        with tf.name_scope('branch_2'):
            branch_2 = conv2d_bn(
                input_tensor,
                64,
                1,
                1,
                padding='same',
                strides=1,
                name=(name + '_3x3_2_reduce'))
            branch_2 = conv2d_bn(
                branch_2,
                96,
                3,
                3,
                padding='same',
                strides=1,
                name=(name + '_3x3_2'))
            branch_2 = conv2d_bn(
                branch_2,
                96,
                3,
                3,
                padding='same',
                strides=1,
                name=(name + '_3x3_3'))
        with tf.name_scope('branch_3'):
            branch_3 = AveragePooling2D(
                pool_size=3, strides=1, padding='same',
                name='_avepool')(input_tensor)
            branch_3 = conv2d_bn(
                branch_3,
                96,
                1,
                1,
                padding='same',
                strides=1,
                name=(name + '_1x1'))
        with tf.name_scope('concat'):
            output = Concatenate(name=(name + '_concat'))(
                [branch_0, branch_1, branch_2, branch_3])
    return output


def reduction_a(input_tensor, name=None):
    with tf.variable_scope(name):
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
                192,
                1,
                1,
                padding='same',
                strides=1,
                name=(name + '_3x3_2_reduce'))
            branch_1 = conv2d_bn(
                branch_1,
                224,
                3,
                3,
                padding='same',
                strides=1,
                name=(name + '_3x3_2'))
            branch_1 = conv2d_bn(
                branch_1,
                256,
                3,
                3,
                padding='valid',
                strides=2,
                name=(name + '_3x3_3'))
        with tf.name_scope('branch_2'):
            branch_2 = MaxPooling2D(
                pool_size=3, strides=2, name=(name + '_pool'))(input_tensor)
        with tf.name_scope('concat'):
            output = Concatenate(name=(name + '_concat'))(
                [branch_0, branch_1, branch_2])
    return output


def inception_b(input_tensor, name=None):
    with tf.variable_scope(name):
        with tf.name_scope('branch_0'):
            branch_0 = conv2d_bn(
                input_tensor,
                384,
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
                7,
                padding='same',
                strides=1,
                name=(name + '_1x7_reduce'))
            branch_1 = conv2d_bn(
                branch_1,
                224,
                7,
                1,
                padding='same',
                strides=1,
                name=(name + '_1x7'))
            branch_1 = conv2d_bn(
                branch_1,
                256,
                3,
                3,
                padding='same',
                strides=1,
                name=(name + '_7x1'))
        with tf.name_scope('branch_2'):
            branch_2 = conv2d_bn(
                input_tensor,
                192,
                1,
                1,
                padding='same',
                strides=1,
                name=(name + '_7x1_2_reduce'))
            branch_2 = conv2d_bn(
                branch_2,
                192,
                7,
                1,
                padding='same',
                strides=1,
                name=(name + '_7x1_2'))
            branch_2 = conv2d_bn(
                branch_2,
                224,
                1,
                7,
                padding='same',
                strides=1,
                name=(name + '_1x7_2'))
            branch_2 = conv2d_bn(
                branch_2,
                224,
                7,
                1,
                padding='same',
                strides=1,
                name=(name + '_7x1_3'))
            branch_2 = conv2d_bn(
                branch_2,
                256,
                1,
                7,
                padding='same',
                strides=1,
                name=(name + '_1x7_3'))
        with tf.name_scope('branch_3'):
            branch_3 = AveragePooling2D(
                pool_size=3, strides=1, padding='same',
                name='_avepool')(input_tensor)
            branch_3 = conv2d_bn(
                branch_3,
                128,
                1,
                1,
                padding='same',
                strides=1,
                name=(name + '_1x1'))
        with tf.name_scope('concat'):
            output = Concatenate(name=(name + '_concat'))(
                [branch_0, branch_1, branch_2, branch_3])
    return output


def reduction_b(input_tensor, name):
    with tf.variable_scope(name):
        with tf.name_scope('branch_0'):
            branch_0 = conv2d_bn(
                input_tensor,
                192,
                1,
                1,
                padding='same',
                strides=1,
                name=(name + '_3x3_reduce'))
            branch_0 = conv2d_bn(
                branch_0,
                192,
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
                padding='same',
                strides=1,
                name=(name + '_1x7_reduce'))
            branch_1 = conv2d_bn(
                branch_1,
                256,
                1,
                7,
                padding='same',
                strides=1,
                name=(name + '_1x7'))
            branch_1 = conv2d_bn(
                branch_1,
                320,
                7,
                1,
                padding='same',
                strides=1,
                name=(name + '_7x1'))
            branch_1 = conv2d_bn(
                branch_1,
                320,
                3,
                3,
                padding='valid',
                strides=2,
                name=(name + '_3x3_2'))
        with tf.name_scope('branch_2'):
            branch_2 = MaxPooling2D(
                pool_size=3, strides=2, name=(name + '_pool'))(input_tensor)
        with tf.name_scope('concat'):
            output = Concatenate(name=(name + '_concat'))(
                [branch_0, branch_1, branch_2])
    return output


def inception_c(input_tensor, name):
    with tf.variable_scope(name):
        with tf.name_scope('branch_0'):
            branch_0 = conv2d_bn(
                input_tensor,
                256,
                1,
                1,
                padding='same',
                strides=1,
                name=(name + '_1x1_2'))
        with tf.name_scope('branch_1'):
            branch_1 = conv2d_bn(
                input_tensor,
                384,
                1,
                1,
                padding='same',
                strides=1,
                name=(name + '_1x1_3'))
            branch_1_1 = conv2d_bn(
                branch_1,
                256,
                1,
                3,
                padding='same',
                strides=1,
                name=(name + '_1x3'))
            branch_1_2 = conv2d_bn(
                branch_1,
                256,
                3,
                1,
                padding='same',
                strides=1,
                name=(name + '_3x1'))
        with tf.name_scope('branch_2'):
            branch_2 = conv2d_bn(
                input_tensor,
                384,
                1,
                1,
                padding='same',
                strides=1,
                name=(name + '_1x1_4'))
            branch_2 = conv2d_bn(
                branch_2,
                448,
                3,
                1,
                padding='same',
                strides=1,
                name=(name + '_3x1_2'))
            branch_2 = conv2d_bn(
                branch_2,
                512,
                1,
                3,
                padding='same',
                strides=1,
                name=(name + '_1x3_2'))
            branch_2_1 = conv2d_bn(
                branch_2,
                256,
                1,
                3,
                padding='same',
                strides=1,
                name=(name + '_1x3_3'))
            branch_2_2 = conv2d_bn(
                branch_2,
                256,
                3,
                1,
                padding='same',
                strides=1,
                name=(name + '_3x1_3'))
        with tf.name_scope('branch_3'):
            branch_3 = AveragePooling2D(
                pool_size=3, strides=1, padding='same',
                name='_avepool')(input_tensor)
            branch_3 = conv2d_bn(
                branch_3,
                256,
                1,
                1,
                padding='same',
                strides=1,
                name=(name + '_1x1'))
        with tf.name_scope('concat'):
            output = Concatenate(name=(name + '_concat'))([
                branch_0, branch_1_1, branch_1_2, branch_2_1, branch_2_2,
                branch_3
            ])
    return output


def inference(images, n_classes):
    
    with tf.name_scope('conv1') as scope:
        conv1 = conv2d_bn(
            images, 32, 3, 3, padding='valid', strides=2, name=scope)

    with tf.name_scope('conv2') as scope:
        conv2 = conv2d_bn(
            conv1, 32, 3, 3, padding='valid', strides=1, name=scope)

    with tf.name_scope('conv3') as scope:
        conv3 = conv2d_bn(
            conv2, 64, 3, 3, padding='same', strides=1, name=scope)

    with tf.name_scope('stem1') as scope:
        with tf.name_scope('branch_0'):
            branch_0 = conv2d_bn(conv3, 96, 3, 3, padding='valid', strides=2)
        with tf.name_scope('branch_1'):
            branch_1 = MaxPooling2D(3, strides=2)(conv3)
        stem1 = Concatenate(name=scope)([branch_0, branch_1])

    with tf.name_scope('stem2') as scope:
        with tf.name_scope('branch_0'):
            branch_0 = conv2d_bn(stem1, 64, 1, 1, padding='valid', strides=1)
            branch_0 = conv2d_bn(
                branch_0, 96, 3, 3, padding='valid', strides=1)
        with tf.name_scope('branch_1'):
            branch_1 = conv2d_bn(stem1, 64, 1, 1, padding='valid', strides=1)
            branch_1 = conv2d_bn(
                branch_1, 64, 1, 7, padding='same', strides=1)
            branch_1 = conv2d_bn(
                branch_1, 64, 7, 1, padding='same', strides=1)
            branch_1 = conv2d_bn(
                branch_1, 96, 3, 3, padding='valid', strides=1)
        stem2 = Concatenate(name=scope)([branch_0, branch_1])

    with tf.name_scope('stem3') as scope:
        with tf.name_scope('branch_0'):
            branch_0 = conv2d_bn(
                stem2, 192, 3, 3, padding='valid', strides=2)
        with tf.name_scope('branch_1'):
            branch_1 = MaxPooling2D(3, strides=2)(stem2)
        net = Concatenate(name=scope)([branch_0, branch_1])

    for i in range(1, 5):
        with tf.name_scope('a' + str(i)) as scope:
            net = inception_a(net, scope)

    with tf.name_scope('a') as scope:
        net = reduction_a(net, scope)

    for i in range(1, 8):
        with tf.name_scope('b' + str(i)) as scope:
            net = inception_b(net, scope)

    with tf.name_scope('b') as scope:
        net = reduction_b(net, scope)

    for i in range(1, 4):
        with tf.name_scope('c' + str(i)) as scope:
            net = inception_c(net, scope)

    with tf.name_scope('pool_8x8') as scope:
        net = GlobalAveragePooling2D(name=scope)(net)

    with tf.name_scope('drop') as scope:
        net = Dropout(rate=0.2)(net)

    with tf.name_scope('classifier') as scope:
        net = Dense(n_classes, activation='softmax', name=scope)(net)

    return net


if __name__ == '__main__':
    images = np.random.randn(128, 299, 299, 3)
    labels = np.random.randn(100)
    images_placeholder = tf.placeholder(tf.float32, [None, 299, 299, 3])
    labels_placeholder = tf.placeholder(tf.float32, [None])
    logits = inference(images_placeholder, 100)
    with tf.Session() as sess:

        tf.global_variables_initializer().run()

        writer = tf.summary.FileWriter("./logs/InceptionV4", sess.graph)
        merged = tf.summary.merge_all()
        #summary = sess.run(merged,feed_dict={images_placeholder: images,labels_placeholder: labels})
        summary = sess.run(merged)
        writer.add_summary(summary)
        writer.close()