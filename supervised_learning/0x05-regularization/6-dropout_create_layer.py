#!/usr/bin/env python3
""" doc """
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """ doc """
    dropped = tf.layers.Dropout(rate=keep_prob)
    init = (tf.contrib.layers.
            variance_scaling_initializer(mode="FAN_AVG"))
    layer = tf.layers.Dense(n, activation, name='layer',
                            kernel_initializer=init,
                            activity_regularizer=dropped)(prev)
    return (layer)
