#!/usr/bin/env python3
"""
training operation for a neural network
in tensorflow using the gradient descent
with momentum optimization algorithm
"""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """ doc """
    obj = tf.train.RMSPropOptimizer(alpha, beta2, epsilon=epsilon)
    return obj.minimize(loss)
