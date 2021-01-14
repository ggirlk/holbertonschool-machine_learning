#!/usr/bin/env python3
"""
training operation for a neural network
in tensorflow using the gradient descent
with momentum optimization algorithm
"""
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """ doc """
    obj = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon=epsilon)
    return obj.minimize(loss)
