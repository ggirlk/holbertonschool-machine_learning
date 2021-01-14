#!/usr/bin/env python3
"""
training operation for a neural network
in tensorflow using the gradient descent
with momentum optimization algorithm
"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """ doc """
    new = tf.train.inverse_time_decay(alpha, global_step, decay_step, decay_rate)
    return new
