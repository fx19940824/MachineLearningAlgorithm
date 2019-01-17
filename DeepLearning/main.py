import numpy as np

import argparse
import os
import sys
import time

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Model

import DataManager
from ModelManager import VGG,Inception

FLAGS = None


def placeholder_inputs(
        batch_size,
        input_size,
):

    images_placeholder = tf.placeholder(
        tf.float32, shape=(batch_size, 224, 224, 3))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))

    return images_placeholder, labels_placeholder


def fill_feed_dict(data_set, images_pl, labels_pl):
    images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size)

    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed,
    }
    return feed_dict


def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set,
            iterator=None):
    true_count = 0
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size

    for step in range(steps_per_epoch):
        #feed_dict = fill_feed_dict(data_set, images_placeholder,
        #                           labels_placeholder)
        feed_dict = {
            images_placeholder: data_set.images,
            labels_placeholder: data_set.labels
        }
        try:
            true_count += sess.run(eval_correct, feed_dict=feed_dict)
        except tf.errors.OutOfRangeError:
            sess.run(iterator.initializer, feed_dict=feed_dict)
            true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = float(true_count) / num_examples
    print('Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
          (num_examples, true_count, precision))


def run_training():
    #(x_train, y_train), (x_test, y_test) = mnist.load_data()
    #data_sets = np.concatenate(x_train, y_train)
    data_sets = input_data.read_data_sets(FLAGS.input_data_dir, one_hot=True)

    #X, y = DataManager.produce_images(data_sets.train.images,
    #                                  data_sets.train.labels, 784, 10,
    #                                  FLAGS.batch_size)

    with tf.Graph().as_default():
        #images_placeholder, labels_placeholder = placeholder_inputs(
        #    FLAGS.batch_size, data_sets.train.images.shape[1])
        sess = tf.Session()
        dataset = tf.data.Dataset.from_tensor_slices((data_sets.train.images,
                                                      data_sets.train.labels))
        dataset = dataset.batch(FLAGS.batch_size)
        iterator = dataset.make_initializable_iterator()
        images_placeholder = tf.placeholder(tf.float32, [None, 784])
        labels_placeholder = tf.placeholder(tf.float32, [None, 10])

        feed_dict = {
            images_placeholder: data_sets.train.images,
            labels_placeholder: data_sets.train.labels
        }

        sess.run(iterator.initializer, feed_dict=feed_dict)
        X, y = iterator.get_next()

        X = tf.reshape(X, shape=(-1, 28, 28, 1))

        logits = VGG.inference(X, 10)

        with tf.name_scope("cost"):
            loss = VGG.loss(logits, y)
            train_op = VGG.training(loss, FLAGS.learning_rate)
            tf.summary.scalar("cost", loss)

        with tf.name_scope("accuracy"):
            acc_op = VGG.evaluation(logits, y)
            tf.summary.scalar("accuray", acc_op)

        summary = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, graph=sess.graph)

        sess.run(init)

        for step in range(FLAGS.max_steps):
            start_time = time.time()

            #feed_dict = fill_feed_dict(data_sets.train, images_placeholder,
            #                           labels_placeholder)

            try:
                _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
            except tf.errors.OutOfRangeError:
                sess.run(iterator.initializer, feed_dict=feed_dict)
                _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

            duration = time.time() - start_time

            if step % 100 == 0:
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value,
                                                           duration))

                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)

                print('Training Data Eval')
                do_eval(sess, acc_op, images_placeholder, labels_placeholder,
                        data_sets.train, iterator)

                print('Validation Data Eval:')
                do_eval(sess, acc_op, images_placeholder, labels_placeholder,
                        data_sets.validation, iterator)

                print('Test Data Eval:')
                do_eval(sess, acc_op, images_placeholder, labels_placeholder,
                        data_sets.test, iterator)


def main(_):
    #delete logs if exit
    os.environ['TF_SYNC_ON_FINISH'] = '0'
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    run_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_data_dir',
        type=str,
        default=os.path.join(os.getcwd(), 'training'),
        help='Directory to put train data')
    parser.add_argument(
        '--test_dir',
        type=str,
        default=os.path.join(os.getcwd(), 'test'),
        help='Directory to put train data')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.00001,
        help='Initial learning rate')
    parser.add_argument(
        '--max_steps',
        type=int,
        default=2000,
        help='Number of steps to run trainer.')
    parser.add_argument(
        '--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument(
        '--log_dir',
        type=str,
        default=os.path.join(os.getcwd(), 'logs'),
        help='Directory to put the log data.')
    parser.add_argument(
        '--num_batches',
        type=int,
        default=100,
        help='Number of batches to run')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)