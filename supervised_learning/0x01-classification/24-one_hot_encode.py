#!/usr/bin/env python3
"""
converts a numeric label
vector into a one-hot matrix
"""
import numpy as np


def one_hot_encode(Y, classes):
    """ doc """
    if type(Y) is not np.ndarray\
       or len(Y) == 0\
       or type(classes) is not int\
       or len(Y.shape) != 1\
       or classes != Y.max()+1:
        return None
    m = Y.shape[0]
    return np.eye(m)[Y].T
