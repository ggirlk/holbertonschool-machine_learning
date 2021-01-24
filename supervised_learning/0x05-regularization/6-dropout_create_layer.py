#!/usr/bin/env python3
""" doc """
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """ doc """
    dropped = tf.nn.dropout(prev, keep_prob=keep_prob)
    init = (tf.contrib.layers.
            variance_scaling_initializer(mode="FAN_AVG"))
    layer = tf.layers.Dense(n, activation, name='layer',
                            kernel_initializer=init)(dropped)
    return layer