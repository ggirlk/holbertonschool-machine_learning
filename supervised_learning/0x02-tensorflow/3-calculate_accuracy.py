#!/usr/bin/env python3
""" doc """
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """ doc """
    acc = tf.compat.v1.metrics.accuracy(y, y_pred)

    return acc
