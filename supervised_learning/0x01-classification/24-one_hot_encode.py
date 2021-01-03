#!/usr/bin/env python3
"""
converts a numeric label
vector into a one-hot matrix
"""
import numpy as np


def one_hot_encode(Y, classes):
    """ doc """
    if Y is None\
       or type(Y) is not np.ndarray\
       or type(classes) is not int\
       or len(Y) == 0\
       or len(Y.shape) != 1\
       or classes-1 != Y.max():
        return None
    mx = np.zeros((classes, Y.shape[0]))
    i = 0
    for n in Y:
        mx[n:n+1, i:i+1] = 1
        i += 1
    return mx
