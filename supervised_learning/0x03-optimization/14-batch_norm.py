#!/usr/bin/env python3
"""
training operation for a neural network
in tensorflow using the gradient descent
with momentum optimization algorithm
"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """ doc """
    layer = tf.layers.Dense(n, activation)(prev)
    return layer
