#!/usr/bin/env python3
""" doc """
import tensorflow as tf


def create_placeholders(nx, classes):
    """ doc """
    x = tf.placeholder("float", [None, nx])
    y = tf.placeholder("float", [None, classes])
    return x, y
