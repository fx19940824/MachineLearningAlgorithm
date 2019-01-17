import numpy as np

import tensorflow as tf
from keras.layers import Conv2D, BatchNormalization, Activation
from keras.layers import MaxPooling2D, AveragePooling2D, Concatenate, Dropout, Flatten, Dense


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
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = BatchNormalization(scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x


def inception_a(X, f1, f5_1, f5_2, f3_1, f3_2, f3_3, fp, name=None):
    if name == None:
        name = 'layer'

    with tf.name_scope(name + '_1x1') as scope:
        branch_1 = conv2d_bn(X, f1, 1, 1, name=scope)

    with tf.name_scope(name + '_5x5') as scope:
        branch_5 = conv2d_bn(X, f5_1, 1, 1, name=(scope + '_reduce'))
        branch_5 = conv2d_bn(branch_5, f5_2, 5, 5, name=scope)

    with tf.name_scope(name + '_3x3') as scope:
        branch_3 = conv2d_bn(X, f3_1, 1, 1, name=(scope + '_reduce'))
        branch_3 = conv2d_bn(branch_3, f3_2, 3, 3, name=(scope + '_1'))
        branch_3 = conv2d_bn(branch_3, f3_3, 3, 3, name=(scope + '_2'))

    with tf.name_scope(name + '_pool') as scope:
        branch_pool = AveragePooling2D(3, strides=1, padding='same')(X)
        branch_pool = conv2d_bn(branch_pool, fp, 1, 1, name=scope)

    with tf.name_scope(name + '_output') as scope:
        output = Concatenate(name=scope)(
            [branch_1, branch_5, branch_3, branch_pool])
    return output


def inception_b(X, f3, f3dbl_1, f3dbl_2, f3dbl_3, name=None):
    if name == None:
        name = 'layer'

    with tf.name_scope(name + '_3x3') as scope:
        branch_3 = conv2d_bn(
            X, f3, 3, 3, strides=(2, 2), padding='valid', name=scope)

    with tf.name_scope(name + '_3x3dbl') as scope:
        branch_3_dbl = conv2d_bn(X, f3dbl_1, 1, 1)
        branch_3_dbl = conv2d_bn(branch_3_dbl, f3dbl_2, 3, 3)
        branch_3_dbl = conv2d_bn(
            branch_3_dbl,
            f3dbl_3,
            3,
            3,
            strides=(2, 2),
            padding='valid',
            name=scope)

    with tf.name_scope(name + '_pool') as scope:
        branch_pool = MaxPooling2D(3, strides=(2, 2))(X)

    with tf.name_scope(name + '_output') as scope:
        merge = Concatenate(name=scope)([branch_3, branch_3_dbl, branch_pool])
    return merge


def inception_c(X,
                f1,
                f7_1,
                f7_2,
                f7_3,
                f7dbl_1,
                f7dbl_2,
                f7dbl_3,
                f7dbl_4,
                f7dbl_5,
                fp,
                name=None):
    if name == None:
        name = 'layer'

    with tf.name_scope(name + '_1x1') as scope:
        branch_1 = conv2d_bn(X, f1, 1, 1, name=scope)

    with tf.name_scope(name + '_7x7') as scope:
        branch_7 = conv2d_bn(X, f7_1, 1, 1)
        branch_7 = conv2d_bn(branch_7, f7_2, 1, 7)
        branch_7 = conv2d_bn(branch_7, f7_3, 7, 1, name=scope)

    with tf.name_scope(name + '_7x7dbl') as scope:
        branch_7_dbl = conv2d_bn(X, f7dbl_1, 1, 1)
        branch_7_dbl = conv2d_bn(branch_7_dbl, f7dbl_2, 7, 1)
        branch_7_dbl = conv2d_bn(branch_7_dbl, f7dbl_3, 1, 7)
        branch_7_dbl = conv2d_bn(branch_7_dbl, f7dbl_4, 7, 1)
        branch_7_dbl = conv2d_bn(branch_7_dbl, f7dbl_5, 1, 7, name=scope)

    with tf.name_scope(name + '_pool') as scope:
        branch_pool = AveragePooling2D(3, strides=1, padding='same')(X)
        branch_pool = conv2d_bn(branch_pool, fp, 1, 1, name=scope)

    with tf.name_scope(name + '_output') as scope:
        output = Concatenate(name=scope)(
            [branch_1, branch_7, branch_7_dbl, branch_pool])
    return output


def inception_d(X, f3_1, f3_2, f773_1, f773_2, f773_3, f773_4, name=None):
    if name == None:
        name = 'layer'

    with tf.name_scope(name + '_3x3') as scope:
        branch_3 = conv2d_bn(X, f3_1, 1, 1)
        branch_3 = conv2d_bn(
            branch_3, f3_2, 3, 3, strides=(2, 2), padding='valid', name=scope)

    with tf.name_scope(name + '_7x7x3') as scope:
        branch_7x7x3 = conv2d_bn(X, f773_1, 1, 1)
        branch_7x7x3 = conv2d_bn(branch_7x7x3, f773_2, 1, 7)
        branch_7x7x3 = conv2d_bn(branch_7x7x3, f773_3, 7, 1)
        branch_7x7x3 = conv2d_bn(
            branch_7x7x3,
            f773_4,
            3,
            3,
            strides=(2, 2),
            padding='valid',
            name=scope)

    with tf.name_scope(name + '_pool') as scope:
        branch_pool = MaxPooling2D(3, strides=(2, 2))(X)

    with tf.name_scope(name + '_output') as scope:
        merge = Concatenate(name=scope)([branch_3, branch_7x7x3, branch_pool])
    return merge


def inception_e(X,
                f1,
                f3_1,
                f3_2,
                f3_3,
                f3dbl_1,
                f3dbl_2,
                f3dbl_3,
                f3dbl_4,
                fp,
                name=None):
    if name == None:
        name = 'layer'

    with tf.name_scope(name + '_1x1') as scope:
        branch_1 = conv2d_bn(X, f1, 1, 1, name=scope)

    with tf.name_scope(name + '_3x3') as scope:
        branch3x3 = conv2d_bn(X, f3_1, 1, 1)
        branch3x3_1 = conv2d_bn(branch3x3, f3_2, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, f3_3, 3, 1)
        branch3x3 = Concatenate(name=scope)([branch3x3_1, branch3x3_2])

    with tf.name_scope(name + '_3x3dbl') as scope:
        branch3x3dbl = conv2d_bn(X, f3dbl_1, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, f3dbl_2, 3, 3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, f3dbl_3, 1, 3)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, f3dbl_4, 3, 1)
        branch3x3dbl = Concatenate(name=scope)(
            [branch3x3dbl_1, branch3x3dbl_2])

    with tf.name_scope(name + '_pool') as scope:
        branch_pool = AveragePooling2D(3, strides=1, padding='same')(X)
        branch_pool = conv2d_bn(branch_pool, fp, 1, 1, name=scope)

    with tf.name_scope(name + '_output') as scope:
        output = Concatenate(name=scope)(
            [branch_1, branch3x3, branch3x3dbl, branch_pool])
    return output


def inference(img_input, n_classes):
    '''input image shape 299,299,3

    '''
    with tf.name_scope('conv1_3x3') as scope:
        x = conv2d_bn(
            img_input, 32, 3, 3, strides=(2, 2), padding='valid', name=scope)

    with tf.name_scope('conv2_3x3') as scope:
        x = conv2d_bn(x, 32, 3, 3, strides=(1, 1), padding='valid', name=scope)

    with tf.name_scope('conv3_3x3') as scope:
        x = conv2d_bn(x, 64, 3, 3, strides=(1, 1), name=scope)
        x = MaxPooling2D(3, strides=2)(x)

    with tf.name_scope('conv4_3x3') as scope:
        x = conv2d_bn(x, 80, 1, 1, padding='valid')
        x = conv2d_bn(x, 192, 3, 3, padding='valid', name=scope)
        x = MaxPooling2D(3, strides=2)(x)

    with tf.name_scope('inception_a1') as scope:
        x = inception_a(x, 64, 48, 64, 64, 96, 96, 32, name=scope)

    with tf.name_scope('inception_a2') as scope:
        x = inception_a(x, 64, 48, 64, 64, 96, 96, 64, name=scope)

    with tf.name_scope('inception_a3') as scope:
        x = inception_a(x, 64, 48, 64, 64, 96, 96, 64, name=scope)

    with tf.name_scope('inception_b') as scope:
        x = inception_b(x, 384, 64, 96, 96, name=scope)

    with tf.name_scope('inception_c1') as scope:
        x = inception_c(
            x, 192, 128, 128, 192, 128, 128, 128, 128, 192, 192, name=scope)

    with tf.name_scope('inception_c2') as scope:
        x = inception_c(
            x, 192, 160, 160, 192, 160, 160, 160, 160, 192, 192, name=scope)

    with tf.name_scope('inception_c3') as scope:
        x = inception_c(
            x, 192, 160, 160, 192, 160, 160, 160, 160, 192, 192, name=scope)

    with tf.name_scope('inception_c4') as scope:
        x = inception_c(
            x, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, name=scope)

    with tf.name_scope('inception_d') as scope:
        x = inception_d(x, 192, 320, 192, 192, 192, 192, name=scope)

    with tf.name_scope('inception_e1') as scope:
        x = inception_e(
            x, 320, 384, 384, 384, 448, 384, 384, 384, 192, name=scope)

    with tf.name_scope('inception_e2') as scope:
        x = inception_e(
            x, 320, 384, 384, 384, 448, 384, 384, 384, 192, name=scope)

    with tf.name_scope('pool_8x8') as scope:
        x = AveragePooling2D(pool_size=8, strides=(1, 1))(x)
        x = Dropout(rate=0.2, name=scope)(x)

    with tf.name_scope('output') as scope:
        x = Flatten()(x)
        output = Dense(n_classes, activation='relu', name=scope)(x)

    return output


def evaluation(logits, labels, k, one_hot=False):
    correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))

    return tf.reduce_sum(tf.cast(correct, tf.float32))


if __name__ == '__main__':
    images = np.random.randn(128, 299, 299, 3)
    labels = np.random.randn(100)
    images_placeholder = tf.placeholder(tf.float32, [None, 299, 299, 3])
    labels_placeholder = tf.placeholder(tf.float32, [None])
    logits = inference(images_placeholder, 100)
    with tf.Session() as sess:

        tf.global_variables_initializer().run()

        writer = tf.summary.FileWriter("./logs/Inception_v3", sess.graph)
        merged = tf.summary.merge_all()
        summary = sess.run(
            merged,
            feed_dict={
                images_placeholder: images
                #labels_placeholder: labels
            })
        writer.add_summary(summary)
        writer.close()
