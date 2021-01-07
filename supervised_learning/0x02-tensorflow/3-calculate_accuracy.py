#!/usr/bin/env python3
""" doc """
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """ doc """
    #acc = tf.compat.v1.metrics.accuracy(y, y_pred)
    acc = tf.math.reduce_sum(y == y_pred)/y.shape[0]
    return tf.cast(acc, tf.float32)
