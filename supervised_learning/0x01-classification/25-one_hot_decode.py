#!/usr/bin/env python3
"""
converts a one-hot matrix
into a vector of labels
"""
import numpy as np


def one_hot_decode(one_hot):
    """ doc """
    if type(one_hot) is not np.ndarray\
       or len(one_hot.shape) != 2:
        return None
    """arr = []
    for i in one_hot.T:
        new = np.where(i == 1)[0]
        if len(new) != 0:
            arr.append(new[0])
    return np.array(arr)"""
    try:
        return np.argmax(one_hot, axis=0)
    except Exception:
        return None
