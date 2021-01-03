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
    return np.array([np.where(i == 1)[0][0]
                     for i in one_hot.T])
