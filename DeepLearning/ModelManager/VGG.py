import numpy as np

import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Model


def VGG_D(input_tensor, input_shape=(224, 224, 3)):
    img_input = input_tensor

    #block1
    x = Conv2D(
        64, 3, activation='relu', padding='same',
        name='block1_conv1')(img_input)
    x = Conv2D(
        64, 3, activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D(2, stride=(2, 2), name='block1_pool')(x)

    #block2
    x = Conv2D(
        128, 3, activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(
        128, 3, activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D(2, stride=(2, 2), name='block2_pool')(x)

    #block3
    x = Conv2D(
        256, 3, activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(
        256, 3, activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(
        256, 3, activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D(2, stride=(2, 2), name='block3_pool')(x)

    #block4
    x = Conv2D(
        512, 3, activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(
        512, 3, activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(
        512, 3, activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D(2, stride=(2, 2), name='block4_pool')(x)

    #block5
    x = Conv2D(
        512, 3, activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(
        512, 3, activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(
        512, 3, activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D(2, stride=(2, 2), name='block5_pool')(x)

    x = Flatten(name='flatten')(x)

    x = Dense(4096, activation='relu', name='fc1')(x)

    x = Dense(4096, activation='relu', name='fc2')(x)

    x = Dense(4006, activation='softmax', name='prediction')(x)

    model = Model(img_input, x, name='vgg19')

    return model


def VGG_E(input_tensor, input_shape=(224, 224, 3)):
    img_input = input_tensor

    #block1
    x = Conv2D(
        64, 3, activation='relu', padding='same',
        name='block1_conv1')(img_input)
    x = Conv2D(
        64, 3, activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D(2, stride=(2, 2), name='block1_pool')(x)

    #block2
    x = Conv2D(
        128, 3, activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(
        128, 3, activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D(2, stride=(2, 2), name='block2_pool')(x)

    #block3
    x = Conv2D(
        256, 3, activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(
        256, 3, activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(
        256, 3, activation='relu', padding='same', name='block3_conv3')(x)
    x = Conv2D(
        256, 3, activation='relu', padding='same', name='block3_conv4')(x)
    x = MaxPooling2D(2, stride=(2, 2), name='block3_pool')(x)

    #block4
    x = Conv2D(
        512, 3, activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(
        512, 3, activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(
        512, 3, activation='relu', padding='same', name='block4_conv3')(x)
    x = Conv2D(
        512, 3, activation='relu', padding='same', name='block4_conv4')(x)
    x = MaxPooling2D(2, stride=(2, 2), name='block4_pool')(x)

    #block5
    x = Conv2D(
        512, 3, activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(
        512, 3, activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(
        512, 3, activation='relu', padding='same', name='block5_conv3')(x)
    x = Conv2D(
        512, 3, activation='relu', padding='same', name='block5_conv4')(x)
    x = MaxPooling2D(2, stride=(2, 2), name='block5_pool')(x)

    x = Flatten(name='flatten')(x)

    x = Dense(4096, activation='relu', name='fc1')(x)

    x = Dense(4096, activation='relu', name='fc2')(x)

    x = Dense(4006, activation='softmax', name='prediction')(x)

    model = Model(img_input, x, name='vgg19')

    return model


def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())


def inference(images, n_classes):
    parameters = []
    with tf.name_scope('block1') as scope_block:
        with tf.name_scope('block1_conv1') as scope:
            kernel = tf.Variable(
                tf.truncated_normal([3, 3, 1, 64],
                                    dtype=tf.float32,
                                    stddev=1e-1),
                name='block1_w1')
            conv = tf.nn.conv2d(
                images, kernel, strides=[1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(
                tf.constant(0.0, shape=[64], dtype=tf.float32),
                trainable=True,
                name='block1_b1')
            bias = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(bias, name=scope)
            print_activations(conv1)
            tf.summary.histogram('block1_w1',kernel)
            tf.summary.histogram('block1_b1',biases)
            parameters += [kernel, biases]

        with tf.name_scope('block1_conv2') as scope:
            kernel = tf.Variable(
                tf.truncated_normal([3, 3, 64, 64],
                                    dtype=tf.float32,
                                    stddev=1e-1),
                name='block1_w2')
            conv = tf.nn.conv2d(
                conv1, kernel, strides=[1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(
                tf.constant(0.0, shape=[64], dtype=tf.float32),
                trainable=True,
                name='block1_b2')
            bias = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(bias, name=scope)
            print_activations(conv2)
            parameters += [kernel, biases]

        with tf.name_scope('block1_maxpooling') as scope:
            pool1 = tf.nn.max_pool(
                conv2,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='VALID',
                name='block1_pool')
            print_activations(pool1)

    with tf.name_scope('block2') as scope_block:
        with tf.name_scope('block2_conv1') as scope:
            kernel = tf.Variable(
                tf.truncated_normal([3, 3, 64, 128],
                                    dtype=tf.float32,
                                    stddev=1e-1),
                name='weights')
            conv = tf.nn.conv2d(
                pool1, kernel, strides=[1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(
                tf.constant(0.0, shape=[128], dtype=tf.float32),
                trainable=True,
                name='biases')
            bias = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(bias, name=scope)
            print_activations(conv1)
            parameters += [kernel, biases]

        with tf.name_scope('block2_conv2') as scope:
            kernel = tf.Variable(
                tf.truncated_normal([3, 3, 128, 128],
                                    dtype=tf.float32,
                                    stddev=1e-1),
                name='weights')
            conv = tf.nn.conv2d(
                conv1, kernel, strides=[1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(
                tf.constant(0.0, shape=[128], dtype=tf.float32),
                trainable=True,
                name='biases')
            bias = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(bias, name=scope)
            print_activations(conv2)
            parameters += [kernel, biases]

        with tf.name_scope('block2_maxpooling') as scope:
            pool2 = tf.nn.max_pool(
                conv2,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='VALID',
                name='block2_pool')
            print_activations(pool2)

    with tf.name_scope('block3') as scope_block:
        with tf.name_scope('block3_conv1') as scope:
            kernel = tf.Variable(
                tf.truncated_normal([3, 3, 128, 256],
                                    dtype=tf.float32,
                                    stddev=1e-1),
                name='weights')
            conv = tf.nn.conv2d(
                pool2, kernel, strides=[1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(
                tf.constant(0.0, shape=[256], dtype=tf.float32),
                trainable=True,
                name='biases')
            bias = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(bias, name=scope)
            print_activations(conv1)
            parameters += [kernel, biases]

        with tf.name_scope('block3_conv2') as scope:
            kernel = tf.Variable(
                tf.truncated_normal([3, 3, 256, 256],
                                    dtype=tf.float32,
                                    stddev=1e-1),
                name='weights')
            conv = tf.nn.conv2d(
                conv1, kernel, strides=[1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(
                tf.constant(0.0, shape=[256], dtype=tf.float32),
                trainable=True,
                name='biases')
            bias = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(bias, name=scope)
            print_activations(conv2)
            parameters += [kernel, biases]

        with tf.name_scope('block3_conv3') as scope:
            kernel = tf.Variable(
                tf.truncated_normal([3, 3, 256, 256],
                                    dtype=tf.float32,
                                    stddev=1e-1),
                name='weights')
            conv = tf.nn.conv2d(
                conv2, kernel, strides=[1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(
                tf.constant(0.0, shape=[256], dtype=tf.float32),
                trainable=True,
                name='biases')
            bias = tf.nn.bias_add(conv, biases)
            conv3 = tf.nn.relu(bias, name=scope)
            print_activations(conv2)
            parameters += [kernel, biases]

        with tf.name_scope('block3_conv4') as scope:
            kernel = tf.Variable(
                tf.truncated_normal([3, 3, 256, 256],
                                    dtype=tf.float32,
                                    stddev=1e-1),
                name='weights')
            conv = tf.nn.conv2d(
                conv3, kernel, strides=[1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(
                tf.constant(0.0, shape=[256], dtype=tf.float32),
                trainable=True,
                name='biases')
            bias = tf.nn.bias_add(conv, biases)
            conv4 = tf.nn.relu(bias, name=scope)
            print_activations(conv2)
            parameters += [kernel, biases]

        with tf.name_scope('block3_maxpooling') as scope:
            pool3 = tf.nn.max_pool(
                conv4,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='VALID',
                name='block3_pool')
            print_activations(pool3)

    with tf.name_scope('block4') as scope_block:
        with tf.name_scope('block4_conv1') as scope:
            kernel = tf.Variable(
                tf.truncated_normal([3, 3, 256, 512],
                                    dtype=tf.float32,
                                    stddev=1e-1),
                name='weights')
            conv = tf.nn.conv2d(
                pool3, kernel, strides=[1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(
                tf.constant(0.0, shape=[512], dtype=tf.float32),
                trainable=True,
                name='biases')
            bias = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(bias, name=scope)
            print_activations(conv1)
            parameters += [kernel, biases]

        with tf.name_scope('block4_conv2') as scope:
            kernel = tf.Variable(
                tf.truncated_normal([3, 3, 512, 512],
                                    dtype=tf.float32,
                                    stddev=1e-1),
                name='weights')
            conv = tf.nn.conv2d(
                conv1, kernel, strides=[1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(
                tf.constant(0.0, shape=[512], dtype=tf.float32),
                trainable=True,
                name='biases')
            bias = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(bias, name=scope)
            print_activations(conv2)
            parameters += [kernel, biases]

        with tf.name_scope('block4_conv3') as scope:
            kernel = tf.Variable(
                tf.truncated_normal([3, 3, 512, 512],
                                    dtype=tf.float32,
                                    stddev=1e-1),
                name='weights')
            conv = tf.nn.conv2d(
                conv2, kernel, strides=[1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(
                tf.constant(0.0, shape=[512], dtype=tf.float32),
                trainable=True,
                name='biases')
            bias = tf.nn.bias_add(conv, biases)
            conv3 = tf.nn.relu(bias, name=scope)
            print_activations(conv3)
            parameters += [kernel, biases]

        with tf.name_scope('block4_conv4') as scope:
            kernel = tf.Variable(
                tf.truncated_normal([3, 3, 512, 512],
                                    dtype=tf.float32,
                                    stddev=1e-1),
                name='weights')
            conv = tf.nn.conv2d(
                conv3, kernel, strides=[1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(
                tf.constant(0.0, shape=[512], dtype=tf.float32),
                trainable=True,
                name='biases')
            bias = tf.nn.bias_add(conv, biases)
            conv4 = tf.nn.relu(bias, name=scope)
            print_activations(conv4)
            parameters += [kernel, biases]

        with tf.name_scope('block4_maxpooling') as scope:
            pool4 = tf.nn.max_pool(
                conv4,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='VALID',
                name='block4_pool')
            print_activations(pool4)

    with tf.name_scope('block5') as scope_block:
        with tf.name_scope('block5_conv1') as scope:
            kernel = tf.Variable(
                tf.truncated_normal([3, 3, 512, 512],
                                    dtype=tf.float32,
                                    stddev=1e-1),
                name='weights')
            conv = tf.nn.conv2d(
                pool4, kernel, strides=[1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(
                tf.constant(0.0, shape=[512], dtype=tf.float32),
                trainable=True,
                name='biases')
            bias = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(bias, name=scope)
            print_activations(conv1)
            parameters += [kernel, biases]

        with tf.name_scope('block5_conv2') as scope:
            kernel = tf.Variable(
                tf.truncated_normal([3, 3, 512, 512],
                                    dtype=tf.float32,
                                    stddev=1e-1),
                name='weights')
            conv = tf.nn.conv2d(
                conv1, kernel, strides=[1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(
                tf.constant(0.0, shape=[512], dtype=tf.float32),
                trainable=True,
                name='biases')
            bias = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(bias, name=scope)
            print_activations(conv2)
            parameters += [kernel, biases]

        with tf.name_scope('block5_conv3') as scope:
            kernel = tf.Variable(
                tf.truncated_normal([3, 3, 512, 512],
                                    dtype=tf.float32,
                                    stddev=1e-1),
                name='weights')
            conv = tf.nn.conv2d(
                conv2, kernel, strides=[1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(
                tf.constant(0.0, shape=[512], dtype=tf.float32),
                trainable=True,
                name='biases')
            bias = tf.nn.bias_add(conv, biases)
            conv3 = tf.nn.relu(bias, name=scope)
            print_activations(conv3)
            parameters += [kernel, biases]

        with tf.name_scope('block5_conv4') as scope:
            kernel = tf.Variable(
                tf.truncated_normal([3, 3, 512, 512],
                                    dtype=tf.float32,
                                    stddev=1e-1),
                name='weights')
            conv = tf.nn.conv2d(
                conv3, kernel, strides=[1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(
                tf.constant(0.0, shape=[512], dtype=tf.float32),
                trainable=True,
                name='biases')
            bias = tf.nn.bias_add(conv, biases)
            conv4 = tf.nn.relu(bias, name=scope)
            print_activations(conv4)
            parameters += [kernel, biases]

        with tf.name_scope('block5_maxpooling') as scope:
            pool5 = tf.nn.max_pool(
                conv4,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='VALID',
                name='block5_pool')
            print_activations(pool5)

    with tf.name_scope('flatten') as scope:
        flatten = tf.contrib.layers.flatten(pool4)

    with tf.name_scope('full_connection1') as scope:
        fc1 = tf.layers.dense(flatten, 4096)
        fc1 = tf.nn.relu(fc1, name=scope)

    with tf.name_scope('full_connection2') as scope:
        fc2 = tf.layers.dense(fc1, 4096)
        fc2 = tf.nn.relu(fc2, name=scope)

    with tf.name_scope('output') as scope:
        fc3 = tf.layers.dense(fc2, n_classes)
        fc3 = tf.nn.softmax(fc3, name=scope)

    return fc3,parameters


def loss(logits, labels):

    labels = tf.to_int64(labels)

    return tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)


def training(loss, learning_rate):
    tf.summary.scalar('loss', loss)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

    global_step = tf.Variable(0, name='global_step', trainable=False)

    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op


def evaluation(logits, labels):
    correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))

    return tf.reduce_sum(tf.cast(correct, tf.float32))
