import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Add, AveragePooling2D, Dropout
from keras.layers import Dense, Concatenate, Flatten


def inception(X, f_1, f_2_reduce, f_2, f_3_reduce, f_3, f_p, name):
    with tf.name_scope(name + '_1_1') as scope:
        inception_1_1 = Conv2D(f_1, 1, activation='relu', name=scope)(X)
    with tf.name_scope(name + '_3_3') as scope:
        inception_3_3_reduce = Conv2D(
            f_2_reduce, 1, activation='relu', name=(name + '_3_3_reduce'))(X)
        inception_3_3 = Conv2D(
            f_2, 3, padding='same', activation='relu',
            name=scope)(inception_3_3_reduce)
    with tf.name_scope(name + '_5_5') as scope:
        inception_5_5_reduce = Conv2D(
            f_3_reduce, 1, activation='relu', name=(name + '_3_3_reduce'))(X)
        inception_5_5 = Conv2D(
            f_3, 3, padding='same', activation='relu',
            name=scope)(inception_5_5_reduce)
    with tf.name_scope(name + '_pool') as scope:
        inception_pool = MaxPooling2D(3, 1, padding='same')(X)
        inception_pool_proj = Conv2D(
            f_p, 1, activation='relu', name=scope)(inception_pool)
    with tf.name_scope(name + '_output') as scope:
        inception_output = Concatenate(name=scope)(
            [inception_1_1, inception_3_3, inception_5_5, inception_pool_proj])
    return inception_output


def loss(logits, labels, one_hot=False):
    labels = tf.to_int64(labels)
    if one_hot == True:
        return tf.losses.softmax_cross_entropy(
            onehot_labels=labels, logits=logits)
    return tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)


def evaluation(logits, labels, k=1):
    correct = tf.equal(tf.argmax(logits, k), tf.argmax(labels, k))
    return tf.reduce_sum(tf.cast(correct, tf.float32))


def middleloss(X, labels, n_classes, name='layer'):
    with tf.name_scope(name + '_ave_pool') as scope:
        loss_ave_pool = AveragePooling2D(pool_size=5, strides=3, name=scope)(X)
    with tf.name_scope(name + '_conv') as scope:
        loss_conv = Conv2D(
            128, 1, activation='relu', name=scope)(loss_ave_pool)
    with tf.name_scope(name + '_fc') as scope:
        loss_fc = Flatten()(loss_conv)
        loss_fc = Dense(1024, activation='relu')(loss_fc)
        loss_fc = Dropout(rate=0.7, name=scope)(loss_fc)
    with tf.name_scope(name + 'classifier') as scope:
        loss_classifier = Dense(
            n_classes, activation='softmax', name=scope)(loss_fc)

    return loss(loss_classifier, labels)


def infer(images, n_classes, labels=None, mode='predict',
          learning_rate=0.0001):
    '''image shape 227,227,3
    training:labels is necessary and mode is 'training'
    predict:mode is 'predict'
    '''
    if mode not in ['training', 'predict']:
        raise ValueError('unknown infer mode')

    with tf.name_scope('block1') as scope_block:
        conv1_7_7 = Conv2D(
            64,
            7,
            strides=2,
            padding='same',
            activation='relu',
            name='conv1_7_7')(images)
        pool1_3_3 = MaxPooling2D(
            3, strides=2, padding='same', name='pool1_3_3')(conv1_7_7)
        pool1_3_3 = BatchNormalization(name='pool1_norm1')(pool1_3_3)

    with tf.name_scope('block2') as scope_block:
        conv2_3_3 = Conv2D(
            64, 1, activation='relu', name='conv2_3_3_reduce')(pool1_3_3)
        conv2_3_3 = Conv2D(
            192, 3, padding='same', activation='relu',
            name='conv2_3_3')(conv2_3_3)
        conv2_3_3 = BatchNormalization(name='conv2_norm')(conv2_3_3)
        pool2_3_3 = MaxPooling2D(
            pool_size=3, strides=2, padding='same',
            name='pool2_3_3')(conv2_3_3)

    with tf.name_scope('block3') as scope_block:
        inception_3a_output = inception(pool2_3_3, 64, 96, 128, 16, 32, 32,
                                        'inception_3a')
        inception_3b_output = inception(inception_3a_output, 128, 128, 192, 32,
                                        96, 64, 'inception_3b')

        with tf.name_scope('pool3_3_3') as scope:
            pool3_3_3 = MaxPooling2D(
                pool_size=3, strides=2, padding='same',
                name=scope)(inception_3b_output)

    with tf.name_scope('block4') as scope_block:
        inception_4a_output = inception(pool3_3_3, 192, 96, 208, 16, 48, 64,
                                        'inception_4a')
        inception_4b_output = inception(inception_4a_output, 160, 112, 224, 24,
                                        64, 64, 'inception_4b')
        inception_4c_output = inception(inception_4b_output, 128, 128, 256, 24,
                                        64, 64, 'inception_4c')
        inception_4d_output = inception(inception_4c_output, 112, 144, 288, 24,
                                        64, 64, 'inception_4d')
        inception_4e_output = inception(inception_4d_output, 256, 160, 320, 32,
                                        128, 128, 'inception_4e')

        if mode == 'training':
            loss1 = middleloss(inception_4a_output, labels, n_classes, 'loss1')
            loss2 = middleloss(inception_4d_output, labels, n_classes, 'loss2')

        with tf.name_scope('pool4_3_3') as scope:
            pool4_3_3 = MaxPooling2D(
                pool_size=3, strides=2, padding='same',
                name=scope)(inception_4e_output)

    with tf.name_scope('block5') as scope_block:
        inception_5a_output = inception(pool4_3_3, 256, 160, 320, 32, 128, 128,
                                        'inception_5a')
        inception_5b_output = inception(inception_5a_output, 384, 192, 384, 48,
                                        128, 128, 'inception_4b')

        with tf.name_scope('pool5_7_7') as scope:
            pool5_7_7 = AveragePooling2D(
                pool_size=7, strides=1,
                name='pool5_7_7_s1')(inception_5b_output)
            pool5_7_7 = Dropout(rate=0.4, name=scope)(pool5_7_7)

    with tf.name_scope('loss3') as scope:
        loss3_flatten = Flatten()(pool5_7_7)
        loss3_classifier = Dense(
            n_classes, activation='softmax', name=scope)(loss3_flatten)
        if mode == 'training':
            loss3 = loss(loss3_classifier, labels)

    if mode == 'predict':
        return loss3_classifier

    loss_total = 0.3 * loss1 + 0.3 * loss2 + loss3
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss_total, global_step=global_step)
    return train_op


def inference(images, labels, n_classes):
    '''image shape 227,227,3
    '''

    with tf.name_scope('block1') as scope_block:
        conv1_7_7 = Conv2D(
            64,
            7,
            strides=2,
            padding='same',
            activation='relu',
            name='conv1_7_7')(images)
        pool1_3_3 = MaxPooling2D(3, strides=2, name='pool1_3_3')(conv1_7_7)
        pool1_3_3 = BatchNormalization(name='pool1_norm1')(pool1_3_3)

    with tf.name_scope('block2') as scope_block:
        conv2_3_3 = Conv2D(
            64, 1, activation='relu', name='conv2_3_3_reduce')(pool1_3_3)
        conv2_3_3 = Conv2D(
            192, 3, activation='relu', name='conv2_3_3')(conv2_3_3)
        conv2_3_3 = BatchNormalization(name='conv2_norm')(conv2_3_3)
        pool2_3_3 = MaxPooling2D(
            pool_size=3, strides=2, name='pool2_3_3')(conv2_3_3)

    with tf.name_scope('block3') as scope_block:
        with tf.name_scope('inception_3a_1_1') as scope:
            inception_3a_1_1 = Conv2D(
                64, 1, activation='relu', name=scope)(pool2_3_3)
        with tf.name_scope('inception_3a_3_3') as scope:
            inception_3a_3_3_reduce = Conv2D(
                96, 1, activation='relu',
                name='inception_3a_3_3_reduce')(pool2_3_3)
            inception_3a_3_3 = Conv2D(
                128, 3, padding='same', activation='relu',
                name=scope)(inception_3a_3_3_reduce)
        with tf.name_scope('inception_3a_5_5') as scope:
            inception_3a_5_5_reduce = Conv2D(
                16, 1, activation='relu',
                name='inception_3a_5_5_reduce')(pool2_3_3)
            inception_3a_5_5 = Conv2D(
                32, 5, padding='same', activation='relu',
                name=scope)(inception_3a_5_5_reduce)
        with tf.name_scope('inception_3a_pool') as scope:
            inception_3a_pool = MaxPooling2D(
                3, strides=1, padding='same',
                name='inception_3a_pool')(pool2_3_3)
            inception_3a_pool_proj = Conv2D(
                32, 1, activation='relu', name=scope)(inception_3a_pool)
        with tf.name_scope('inception_3a_output') as scope:
            inception_3a_output = Add(name=scope)([
                inception_3a_1_1, inception_3a_3_3, inception_3a_5_5,
                inception_3a_pool_proj
            ])

        with tf.name_scope('inception_3b_1_1') as scope:
            inception_3b_1_1 = Conv2D(
                128, 1, activation='relu', name=scope)(inception_3a_output)
        with tf.name_scope('inception_3b_3_3') as scope:
            inception_3b_3_3_reduce = Conv2D(
                128, 1, activation='relu',
                name='inception_3b_3_3_reduce')(inception_3a_output)
            inception_3b_3_3 = Conv2D(
                192, 3, padding='same', activation='relu',
                name=scope)(inception_3b_3_3_reduce)
        with tf.name_scope('inception_3b_5_5') as scope:
            inception_3b_5_5_reduce = Conv2D(
                32, 1, activation='relu',
                name='inception_3b_5_5_reduce')(inception_3a_output)
            inception_3b_5_5 = Conv2D(
                96, 5, padding='same', activation='relu',
                name=scope)(inception_3b_5_5_reduce)
        with tf.name_scope('inception_3b_pool') as scope:
            inception_3b_pool = MaxPooling2D(
                3, strides=1, padding='same',
                name='inception_3b_pool')(inception_3a_output)
            inception_3b_pool_proj = Conv2D(
                64, 1, activation='relu', name=scope)(inception_3b_pool)
        with tf.name_scope('inception_3b_output') as scope:
            inception_3b_output = Add(name=scope)([
                inception_3b_1_1, inception_3b_3_3, inception_3b_5_5,
                inception_3b_pool_proj
            ])
        with tf.name_scope('pool3_3_3') as scope:
            pool3_3_3 = MaxPooling2D(
                pool_size=3, strides=2, name=scope)(inceoption_3b_output)

    with tf.name_scope('block4') as scope_block:
        with tf.name_scope('inception_4a_1_1') as scope:
            inception_4a_1_1 = Conv2D(
                192, 1, activation='relu', name=scope)(pool3_3_3)
        with tf.name_scope('inception_4a_3_3') as scope:
            inception_4a_3_3_reduce = Conv2D(
                96, 1, activation='relu',
                name='inception_4a_3_3_reduce')(pool3_3_3)
            inception_4a_3_3 = Conv2D(
                208, 3, padding='same', activation='relu',
                name=scope)(inception_4a_3_3_reduce)
        with tf.name_scope('inception_4a_5_5') as scope:
            inception_4a_5_5_reduce = Conv2D(
                16, 1, activation='relu',
                name='inception_4a_5_5_reduce')(pool3_3_3)
            inception_4a_5_5 = Conv2D(
                48, 5, padding='same', activation='relu',
                name=scope)(inception_4a_5_5_reduce)
        with tf.name_scope('inception_4a_pool') as scope:
            inception_4a_pool = MaxPooling2D(
                3, strides=1, padding='same',
                name='inception_4a_pool')(pool3_3_3)
            inception_4a_pool_proj = Conv2D(
                64, 1, activation='relu', name=scope)(inception_4a_pool)
        with tf.name_scope('inception_4a_output') as scope:
            inception_4a_output = Add(name=scope)([
                inception_4a_1_1, inception_4a_3_3, inception_4a_5_5,
                inception_4a_pool_proj
            ])

        with tf.name_scope('loss1') as scope:
            loss1_ave_pool = AveragePooling2D(
                pool_size=5, strides=3,
                name='loss1_ave_pool')(inception_4a_output)
            loss1_conv = Conv2D(
                128, 1, activation='relu', name='loss1_conv')(loss1_ave_pool)
            loss1_fc = Dense(1024, activation='relu')(loss1_conv)
            loss1_fc = Dropout(rate=0.7)(loss1_fc)
            loss1_classifier = Dense(
                n_classes, activation='softmax', name=scope)(loss1_fc)

        with tf.name_scope('inception_4b_1_1') as scope:
            inception_4b_1_1 = Conv2D(
                160, 1, activation='relu', name=scope)(inception_4a_output)
        with tf.name_scope('inception_4b_3_3') as scope:
            inception_4b_3_3_reduce = Conv2D(
                112, 1, activation='relu',
                name='inception_4b_3_3_reduce')(inception_4a_output)
            inception_4b_3_3 = Conv2D(
                224, 3, padding='same', activation='relu',
                name=scope)(inception_4b_3_3_reduce)
        with tf.name_scope('inception_4b_5_5') as scope:
            inception_4b_5_5_reduce = Conv2D(
                24, 1, activation='relu',
                name='inception_4b_5_5_reduce')(inception_4a_output)
            inception_4b_5_5 = Conv2D(
                64, 5, padding='same', activation='relu',
                name=scope)(inception_4b_5_5_reduce)
        with tf.name_scope('inception_4b_pool') as scope:
            inception_4b_pool = MaxPooling2D(
                3, strides=1, padding='same',
                name='inception_4b_pool')(inception_4a_output)
            inception_4b_pool_proj = Conv2D(
                64, 1, activation='relu', name=scope)(inception_4b_pool)
        with tf.name_scope('inception_4b_output') as scope:
            inception_4b_output = Add(name=scope)([
                inception_4b_1_1, inception_4b_3_3, inception_4b_5_5,
                inception_4b_pool_proj
            ])

        with tf.name_scope('inception_4c_1_1') as scope:
            inception_4c_1_1 = Conv2D(
                128, 1, activation='relu', name=scope)(inception_4b_output)
        with tf.name_scope('inception_4c_3_3') as scope:
            inception_4c_3_3_reduce = Conv2D(
                128, 1, activation='relu',
                name='inception_4c_3_3_reduce')(inception_4b_output)
            inception_4c_3_3 = Conv2D(
                256, 3, padding='same', activation='relu',
                name=scope)(inception_4c_3_3_reduce)
        with tf.name_scope('inception_4c_5_5') as scope:
            inception_4c_5_5_reduce = Conv2D(
                24, 1, activation='relu',
                name='inception_4c_5_5_reduce')(inception_4b_output)
            inception_4c_5_5 = Conv2D(
                64, 5, padding='same', activation='relu',
                name=scope)(inception_4c_5_5_reduce)
        with tf.name_scope('inception_4c_pool') as scope:
            inception_4c_pool = MaxPooling2D(
                3, strides=1, padding='same',
                name='inception_4c_pool')(inception_4b_output)
            inception_4c_pool_proj = Conv2D(
                64, 1, activation='relu', name=scope)(inception_4c_pool)
        with tf.name_scope('inception_4c_output') as scope:
            inception_4c_output = Add(name=scope)([
                inception_4c_1_1, inception_4c_3_3, inception_4c_5_5,
                inception_4c_pool_proj
            ])

        with tf.name_scope('inception_4d_1_1') as scope:
            inception_4d_1_1 = Conv2D(
                112, 1, activation='relu', name=scope)(inception_4c_output)
        with tf.name_scope('inception_4d_3_3') as scope:
            inception_4d_3_3_reduce = Conv2D(
                144, 1, activation='relu',
                name='inception_4d_3_3_reduce')(inception_4c_output)
            inception_4d_3_3 = Conv2D(
                288, 3, padding='same', activation='relu',
                name=scope)(inception_4d_3_3_reduce)
        with tf.name_scope('inception_4d_5_5') as scope:
            inception_4d_5_5_reduce = Conv2D(
                32, 1, activation='relu',
                name='inception_4d_5_5_reduce')(inception_4c_output)
            inception_4d_5_5 = Conv2D(
                64, 5, padding='same', activation='relu',
                name=scope)(inception_4d_5_5_reduce)
        with tf.name_scope('inception_4d_pool') as scope:
            inception_4d_pool = MaxPooling2D(
                3, strides=1, padding='same',
                name='inception_4d_pool')(inception_4c_output)
            inception_4d_pool_proj = Conv2D(
                64, 1, activation='relu', name=scope)(inception_4d_pool)
        with tf.name_scope('inception_4d_output') as scope:
            inception_4d_output = Add(name=scope)([
                inception_4d_1_1, inception_4d_3_3, inception_4d_5_5,
                inception_4d_pool_proj
            ])

        with tf.name_scope('loss2') as scope:
            loss2_ave_pool = AveragePooling2D(
                pool_size=5, strides=3,
                name='loss1_ave_pool')(inception_4d_output)
            loss2_conv = Conv2D(
                128, 1, activation='relu', name='loss1_conv')(loss2_ave_pool)
            loss2_fc = Dense(1024, activation='relu')(loss2_conv)
            loss2_fc = Dropout(rate=0.7)(loss2_fc)
            loss2_classifier = Dense(
                n_classes, activation='softmax', name=scope)(loss2_fc)

        with tf.name_scope('inception_4e_1_1') as scope:
            inception_4e_1_1 = Conv2D(
                256, 1, activation='relu', name=scope)(inception_4d_output)
        with tf.name_scope('inception_4e_3_3') as scope:
            inception_4e_3_3_reduce = Conv2D(
                160, 1, activation='relu',
                name='inception_4e_3_3_reduce')(inception_4d_output)
            inception_4e_3_3 = Conv2D(
                320, 3, padding='same', activation='relu',
                name=scope)(inception_4e_3_3_reduce)
        with tf.name_scope('inception_4e_5_5') as scope:
            inception_4e_5_5_reduce = Conv2D(
                32, 1, activation='relu',
                name='inception_4e_5_5_reduce')(inception_4d_output)
            inception_4e_5_5 = Conv2D(
                128, 5, padding='same', activation='relu',
                name=scope)(inception_4e_5_5_reduce)
        with tf.name_scope('inception_4e_pool') as scope:
            inception_4e_pool = MaxPooling2D(
                3, strides=1, padding='same',
                name='inception_4e_pool')(inception_4d_output)
            inception_4e_pool_proj = Conv2D(
                128, 1, activation='relu', name=scope)(inception_4e_pool)
        with tf.name_scope('inception_4e_output') as scope:
            inception_4e_output = Add(name=scope)([
                inception_4e_1_1, inception_4e_3_3, inception_4e_5_5,
                inception_4e_pool_proj
            ])

        with tf.name_scope('pool4_3_3') as scope:
            pool4_3_3 = MaxPooling2D(
                pool_size=3, strides=2, name=scope)(inceoption_4e_output)

    with tf.name_scope('block5') as scope_block:
        with tf.name_scope('inception_5a_1_1') as scope:
            inception_5a_1_1 = Conv2D(
                256, 1, activation='relu', name=scope)(pool4_3_3)
        with tf.name_scope('inception_5a_3_3') as scope:
            inception_5a_3_3_reduce = Conv2D(
                160, 1, activation='relu',
                name='inception_5a_3_3_reduce')(pool4_3_3)
            inception_5a_3_3 = Conv2D(
                320, 3, padding='same', activation='relu',
                name=scope)(inception_5a_3_3_reduce)
        with tf.name_scope('inception_5a_5_5') as scope:
            inception_5a_5_5_reduce = Conv2D(
                32, 1, activation='relu',
                name='inception_5a_5_5_reduce')(pool4_3_3)
            inception_5a_5_5 = Conv2D(
                128, 5, padding='same', activation='relu',
                name=scope)(inception_5a_5_5_reduce)
        with tf.name_scope('inception_5a_pool') as scope:
            inception_5a_pool = MaxPooling2D(
                3, strides=1, padding='same',
                name='inception_5a_pool')(pool4_3_3)
            inception_5a_pool_proj = Conv2D(
                128, 1, activation='relu', name=scope)(inception_5a_pool)
        with tf.name_scope('inception_5a_output') as scope:
            inception_5a_output = Add(name=scope)([
                inception_5a_1_1, inception_5a_3_3, inception_5a_5_5,
                inception_5a_pool_proj
            ])

        with tf.name_scope('inception_5b_1_1') as scope:
            inception_5b_1_1 = Conv2D(
                384, 1, activation='relu', name=scope)(inception_5a_output)
        with tf.name_scope('inception_5b_3_3') as scope:
            inception_5b_3_3_reduce = Conv2D(
                192, 1, activation='relu',
                name='inception_5b_3_3_reduce')(inception_5a_output)
            inception_5b_3_3 = Conv2D(
                384, 3, padding='same', activation='relu',
                name=scope)(inception_5b_3_3_reduce)
        with tf.name_scope('inception_5b_5_5') as scope:
            inception_5b_5_5_reduce = Conv2D(
                48, 1, activation='relu',
                name='inception_5b_5_5_reduce')(inception_5a_output)
            inception_5b_5_5 = Conv2D(
                128, 5, padding='same', activation='relu',
                name=scope)(inception_5b_5_5_reduce)
        with tf.name_scope('inception_5b_pool') as scope:
            inception_5b_pool = MaxPooling2D(
                3, strides=1, padding='same',
                name='inception_5b_pool')(inception_5a_output)
            inception_5b_pool_proj = Conv2D(
                128, 1, activation='relu', name=scope)(inception_5b_pool)
        with tf.name_scope('inception_5b_output') as scope:
            inception_5b_output = Add(name=scope)([
                inception_5b_1_1, inception_5b_3_3, inception_5b_5_5,
                inception_5b_pool_proj
            ])

        with tf.name_scope('pool5_7_7') as scope:
            pool5_7_7 = AveragePooling2D(
                pool_size=7, strides=1,
                name='pool5_7_7_s1')(inception_5b_output)
            pool5_7_7 = Dropout(rate=0.4, name=scope)(pool5_7_7)

    with tf.name_scope('loss3') as scope:
        loss3_classifier = Dense(
            n_classes, activation='softmax', name=scope)(pool5_7_7)


if __name__ == '__main__':
    images = np.random.randn(128, 227, 227, 3)
    labels = np.random.randn(100)
    images_placeholder = tf.placeholder(tf.float32, [None, 227, 227, 3])
    labels_placeholder = tf.placeholder(tf.float32, [None])
    train_op = infer(
        images_placeholder, 1000, labels_placeholder, mode='training')
    with tf.Session() as sess:

        tf.global_variables_initializer().run()

        writer = tf.summary.FileWriter("./logs/GoogleNet_logs", sess.graph)
        merged = tf.summary.merge_all()
        summary = sess.run(
            merged,
            feed_dict={
                images_placeholder: images,
                labels_placeholder: labels
            })
        writer.add_summary(summary)
        writer.close()
