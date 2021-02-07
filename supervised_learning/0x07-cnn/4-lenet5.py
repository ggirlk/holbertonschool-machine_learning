#!/usr/bin/env python3
""" doc """
import tensorflow as tf


def lenet5(x, y):
    """ doc """
    init = tf.contrib.layers.variance_scaling_initializer()
    layer = tf.layers.Conv2D(6, 5, padding='same',
                             activation='relu',
                             kernel_initializer=init,
                             )(x)
    layer = tf.layers.MaxPooling2D(2, 2)(layer)
    layer = tf.layers.Conv2D(16, 5, padding='same',
                             activation='relu',
                             kernel_initializer=init,
                             )(layer)
    layer = tf.layers.MaxPooling2D(2, 2)(layer)
    layer = tf.layers.Flatten()(layer)
    layer = tf.layers.Dense(120, activation='relu',
                            kernel_initializer=init)(layer)
    layer = tf.layers.Dense(84, activation='relu',
                            kernel_initializer=init)(layer)
    layer = tf.layers.Dense(10, activation='softmax',
                            kernel_initializer=init)(layer)
    loss = tf.losses.softmax_cross_entropy(y, layer)
    train_op_adamOpt = tf.train.AdamOptimizer().minimize(loss)
    max_pred = tf.argmax(layer, 1)
    equal = tf.equal(tf.argmax(y, 1), max_pred)
    acc = tf.reduce_mean(tf.cast(equal, tf.float32))
    return layer, train_op_adamOpt, loss, acc
