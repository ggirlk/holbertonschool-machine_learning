#!/usr/bin/env python3
""" doc """
import tensorflow as tf


def create_train_op(loss, alpha):
    """ doc """
    gr = tf.train.GradientDescentOptimizer(alpha)
    return gr.minimize(loss)
