#!/usr/bin/env python3
"""
converts a numeric label
vector into a one-hot matrix
"""
import numpy as np


def one_hot_encode(Y, classes):
    """ doc """
    if type(Y) != np.ndarray\
       or type(classes) is not int\
       or classes < 0\
       or len(Y) == 0\
       or len(Y.shape) != 1\
       or classes-1 != Y.max():
        return None
    m = Y.shape[0]
    mx = np.zeros((classes, m))
    i = 0
    for n in Y:
        if type(n) is not np.int64 or n < 0:
            return None
        mx[n:n+1, i:i+1] = 1
        i += 1
    return mx
