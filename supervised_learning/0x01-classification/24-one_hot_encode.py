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
       or type(classes) is not int:
        return None
    try:
        return np.eye(classes)[Y].T
    except Exception:
        return None
