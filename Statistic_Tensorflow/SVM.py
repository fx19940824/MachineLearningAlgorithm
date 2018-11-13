import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data


def init_weights(shape,name):
    return tf.Variable(tf.random_normal(shape, stddev=0.1),name=name)


def model(X, w, b):
    return tf.subtract(tf.matmul(X, w), b)


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

X = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='train_x')
Y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='train_y')

w = init_weights(shape=[784, 10],name='W')
b = init_weights(shape=[10],name='b')

py_x = model(X, w, b)

classification_term = tf.reduce_mean(
    tf.maximum(0, tf.subtract(1, tf.multiply(py_x, Y))))
alpha = tf.constant([0.01])
l2_norm = tf.reduce_sum(tf.square(X))

loss = tf.add(classification_term, tf.multiply(l2_norm, alpha))

train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

predict_op = tf.argmax(py_x, 1)

with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()

    for i in range(100):
        for start, end in zip(
                range(0, len(trX), 128), range(128,
                                               len(trX) + 1, 128)):
            sess.run(
                train_op, feed_dict={
                    X: trX[start:end],
                    Y: trY[start:end]
                })
        print(
            i,
            np.mean(
                np.argmax(teY, axis=1) == sess.run(
                    predict_op, feed_dict={X: teX})))
