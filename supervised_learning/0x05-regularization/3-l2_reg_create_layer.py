#!/usr/bin/env python3
""" doc """
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """ doc """

    init = (tf.contrib.layers.
            variance_scaling_initializer(mode="FAN_AVG"))
    freg = tf.contrib.layers.l2_regularizer(lambtha)
    layer = tf.layers.Dense(n, activation, name='layer',
                            kernel_initializer=init,
                            kernel_regularizer=freg)(prev)
    return (layer)
