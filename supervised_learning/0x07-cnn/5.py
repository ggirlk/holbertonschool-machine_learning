#!/usr/bin/env python3

import tensorflow as tf
import numpy as np

lenet5 = __import__('4-lenet5').lenet5

np.random.seed(8)
m = np.random.randint(100, 200)
h, w = np.random.randint(20, 50, 2).tolist()

X = np.random.uniform(0, 1, (m, h, w, 1))
Y = np.random.randint(0, 10, m)

tf.set_random_seed(0)
x = tf.placeholder(tf.float32, (None, h, w, 1))
y = tf.placeholder(tf.int32, (None,))
y_oh = tf.one_hot(y, 10)
y_pred, train_op, loss, acc = lenet5(x, y_oh)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for _ in range(10):
            print(sess.run([loss, acc, y_pred], feed_dict={x:X, y:Y}))
            sess.run(train_op, feed_dict={x:X, y:Y})
    print(sess.run([loss, acc, y_pred], feed_dict={x:X, y:Y}))
