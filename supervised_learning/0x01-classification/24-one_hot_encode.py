#!/usr/bin/env python3
"""
converts a numeric label
vector into a one-hot matrix
"""
import numpy as np


def one_hot_encode(Y, classes):
    """ doc """
    if classes is None:
        classes = Y.max()+1
    if type(Y) is not np.ndarray\
       or len(Y) < 1\
       or type(classes) is not int\
       or len(Y.shape) != 1\
       or classes != Y.max()+1:
        return None
    return np.eye(classes)[Y].T
