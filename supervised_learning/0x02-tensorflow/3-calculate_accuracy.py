#!/usr/bin/env python3
""" doc """
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """ doc """
    m = tf.keras.metrics.Accuracy()
    m.update_state(y, y_pred, sample_weight=None)

    return tf.Tensor(m.result().numpy(), shape=(), dtype=float32)
