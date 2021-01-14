#!/usr/bin/env python3
"""
training operation for a neural network
in tensorflow using the gradient descent
with momentum optimization algorithm
"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """ doc """
    init = (tf.contrib.layers.
            variance_scaling_initializer(mode="FAN_AVG"))
    layer = tf.layers.Dense(n, activation,
                            kernel_initializer=init)(prev)
    batch = tf.layers.BatchNormalization(epsilon=1e-8,
                                         beta_initializer=0,
                                         gamma_initializer=1,
                                         )
    output = batch.compute_mask(layer)
