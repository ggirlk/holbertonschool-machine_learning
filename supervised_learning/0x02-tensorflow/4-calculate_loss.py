#!/usr/bin/env python3
""" doc """
import tensorflow as tf


def calculate_loss(y, y_pred):
    """ doc """
    return tf.compat.v1.losses.mean_squared_error(
        y, y_pred, weights=1.0, scope=None,
        loss_collection=tf.GraphKeys.LOSSES, reduction=Reduction.SUM_BY_NONZERO_WEIGHTS
    )
