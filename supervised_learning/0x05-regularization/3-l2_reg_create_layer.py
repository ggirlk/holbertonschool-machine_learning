#!/usr/bin/env python3
""" doc """
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """ doc """

    init = (tf.contrib.layers.
            variance_scaling_initializer(mode="FAN_AVG"))
    layer = tf.layers.Dense(n, name='layer',
                            kernel_regularizer=lambtha,
                            kernel_initializer=init)(prev)
    return activation(layer)
