#!/usr/bin/env python3
import tensorflow as tf


def create_placeholders(nx, classes):
    """ doc """
    x = tf.placeholder("float", None)
    y = x * 2
    return x, y
