#!/usr/bin/env python3
""" doc """
import tensorflow as tf


def create_layer(prev, n, activation):
    """ doc """
    init = (tf.contrib.layers.
                   variance_scaling_initializer(mode="FAN_AVG"))
    layer = tf.layers.Dense(n, activation, name='layer',
                           kernel_initializer=init)(prev)
    return layer

