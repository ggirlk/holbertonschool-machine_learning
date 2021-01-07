#!/usr/bin/env python3
""" doc """
import tensorflow as tf


def calculate_loss(y, y_pred):
    """ doc """
    return tf.losses.softmax_cross_entropy(y, y_pred)
