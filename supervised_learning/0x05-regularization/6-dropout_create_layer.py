#!/usr/bin/env python3
""" doc """
import numpy as np


def dropout_create_layer(prev, n, activation, keep_prob):
    """ doc """
    A = activation(prev)
    d = np.random.rand(A.shape[1], n) < keep_prob
    d = np.where(d < keep_prob, 0, 1)
    layer = np.multiply(A, d) / keep_prob
    return layer