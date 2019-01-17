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


def conv_block(input_tensor, kernel_size, filters, strides=(2, 2), name=None):
    filters1, filters2, filters3 = filters
    #conv_name_base='res'+str(stage)+block+'_branch'
    #bn_name_base='bn'+str(stage)+block+'_branch'

    with tf.name_scope(name + '_2a') as scope:
        branch = conv_bn(
            input_tensor,
            1,
            filters1,
            padding='same',
            strides=strides,
            name=scope)

    with tf.name_scope(name + '_2b') as scope:
        branch = conv_bn(
            branch,
            kernel_size,
            filters2,
            padding='same',
            strides=1,
            name=scope)

    with tf.name_scope(name + '_2c') as scope:
        branch = Conv2D(
            filters3,
            1,
            kernel_initializer='he_normal',
            padding='same',
            name=scope + '_conv')(branch)
        branch = BatchNormalization(name=scope + '_bn')(branch)

    with tf.name_scope(name + '_shortcut') as scope:
        shortcut = Conv2D(
            filters3, 1, strides=strides, name=scope + '_conv')(input_tensor)
        shortcut = BatchNormalization(name=scope + '_bn')(shortcut)

    output_tensor = Add(name=name)([branch, shortcut])
    output_tensor = Activation('relu')(output_tensor)
    return output_tensor


def identity_block(input_tensor, kernel_size, filters, name):
    filters1, filters2, filters3 = filters

    with tf.name_scope(name + '_2a') as scope:
        branch = conv_bn(input_tensor, 1, filters1, padding='same', name=scope)

    with tf.name_scope(name + '_2b') as scope:
        branch = conv_bn(
            branch,
            kernel_size,
            filters2,
            padding='same',
            strides=1,
            name=scope)

    with tf.name_scope(name + '_2c') as scope:
        branch = Conv2D(
            filters3,
            1,
            kernel_initializer='he_normal',
            padding='same',
            name=scope + '_conv')(branch)
        branch = BatchNormalization(name=scope + '_bn')(branch)

    output_tensor = Add(name=scope)([branch, input_tensor])
    output_tensor = Activation('relu')(output_tensor)
    return output_tensor


def inference(images, n_classes):
    '''input shape 224,224,3
    '''
    with tf.name_scope('conv1') as scope:
        conv1 = conv_bn(images, 7, 64, strides=2, name=scope)

    with tf.name_scope('pool1') as scope:
        pool1 = MaxPooling2D(3, strides=2)(conv1)

    with tf.name_scope('res2') as scope_block:
        with tf.name_scope(scope_block + 'a') as scope:
            res2a = conv_block(pool1, 3, [64, 64, 256], strides=1, name=scope)

        with tf.name_scope(scope_block + 'b') as scope:
            res2b = identity_block(res2a, 3, [64, 64, 256], name=scope)

        with tf.name_scope(scope_block + 'c') as scope:
            res2 = identity_block(res2b, 3, [64, 64, 256], name=scope)

    with tf.name_scope('res3') as scope_block:
        with tf.name_scope(scope_block + 'a') as scope:
            res3a = conv_block(res2, 3, [128, 128, 512], name=scope)

        with tf.name_scope(scope_block + 'b') as scope:
            res3b = identity_block(res3a, 3, [128, 128, 512], name=scope)

        with tf.name_scope(scope_block + 'c') as scope:
            res3c = identity_block(res3b, 3, [128, 128, 512], name=scope)

        with tf.name_scope(scope_block + 'd') as scope:
            res3 = identity_block(res3c, 3, [128, 128, 512], name=scope)

    with tf.name_scope('res4') as scope_block:
        with tf.name_scope(scope_block + 'a') as scope:
            res4a = conv_block(res3, 3, [256, 256, 1024], name=scope)

        with tf.name_scope(scope_block + 'b') as scope:
            res4b = identity_block(res4a, 3, [256, 256, 1024], name=scope)

        with tf.name_scope(scope_block + 'c') as scope:
            res4c = identity_block(res4b, 3, [256, 256, 1024], name=scope)

        with tf.name_scope(scope_block + 'd') as scope:
            res4d = identity_block(res4c, 3, [256, 256, 1024], name=scope)

        with tf.name_scope(scope_block + 'e') as scope:
            res4e = identity_block(res4d, 3, [256, 256, 1024], name=scope)

        with tf.name_scope(scope_block + 'f') as scope:
            res4 = identity_block(res4e, 3, [256, 256, 1024], name=scope)

    with tf.name_scope('res5') as scope_block:
        with tf.name_scope(scope_block + 'a') as scope:
            res5a = conv_block(res4, 3, [512, 512, 2048], name=scope)

        with tf.name_scope(scope_block + 'b') as scope:
            res5b = identity_block(res5a, 3, [512, 512, 2048], name=scope)

        with tf.name_scope(scope_block + 'c') as scope:
            res5 = identity_block(res5b, 3, [512, 512, 2048], name=scope)

    with tf.name_scope('avg_pool') as scope:
        avg_pool = GlobalAveragePooling2D(name=scope)(res5)

    with tf.name_scope('fc') as scope:
        logits = Dense(n_classes, activation='softmax', name=scope)(avg_pool)

    return logits


if __name__ == '__main__':
    images = np.random.randn(128, 224, 224, 3)
    labels = np.random.randn(100)
    images_placeholder = tf.placeholder(tf.float32, [None, 224, 224, 3])
    labels_placeholder = tf.placeholder(tf.float32, [None])
    logits = inference(images_placeholder, 100)
    with tf.Session() as sess:

        tf.global_variables_initializer().run()

        writer = tf.summary.FileWriter("./logs/ResNet50", sess.graph)
        merged = tf.summary.merge_all()
        #summary = sess.run(merged,feed_dict={images_placeholder: images,labels_placeholder: labels})
        summary = sess.run(merged)
        writer.add_summary(summary)
        writer.close()
