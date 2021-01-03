#!/usr/bin/env python3
"""
converts a numeric label
vector into a one-hot matrix
"""
import numpy as np


def one_hot_encode(Y, classes):
    """ doc """
    if type(Y) is not np.ndarray\
       or type(classes) is not int\
       or classes <= 0\
       or len(Y) == 0\
       or len(Y.shape) != 1\
       or classes-1 != Y.max():
        return None
    if classes is None:
        classes = Y.max()+1
    return np.eye(classes)[Y].T
