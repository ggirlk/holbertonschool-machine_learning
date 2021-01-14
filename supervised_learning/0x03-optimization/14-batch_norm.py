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
    m, v = tf.nn.moments(layer, axes=[0])
    beta = tf.Variable(tf.zeros([n]))
    gamma = tf.Variable(tf.ones([n]))
    epsilon = 1e-8,
    scale = None
    output = tf.nn.batch_norm_with_global_normalization(layer,
                                                        m, v,
                                                        beta,
                                                        gamma,
                                                        epsilon,
                                                        scale)
    return output