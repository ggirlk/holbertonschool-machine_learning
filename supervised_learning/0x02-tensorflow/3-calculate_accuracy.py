#!/usr/bin/env python3
""" doc """
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """ doc """
    pred = tf.argmax(y_pred, 1)
    eq = tf.equal(tf.argmax(y, 1), pred)
    return tf.reduce_mean(tf.cast(eq, tf.float32))
