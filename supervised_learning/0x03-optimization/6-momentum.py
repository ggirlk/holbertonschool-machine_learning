#!/usr/bin/env python3
"""
training operation for a neural network
in tensorflow using the gradient descent
with momentum optimization algorithm
"""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """ doc """
    obj = tf.train.MomentumOptimizer(alpha, beta1)
    return obj.minimize(loss)
