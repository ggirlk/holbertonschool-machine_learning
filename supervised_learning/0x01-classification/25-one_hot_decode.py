#!/usr/bin/env python3
"""
converts a one-hot matrix
into a vector of labels
"""
import numpy as np


def one_hot_decode(one_hot):
    if type(one_hot) is not np.ndarray\
       len(one_hot.shape) != 2:
        return None
    m = one_hot.shape[0]
    mx = np.array([0]*m)
    i = 0
    for i in range(len(one_hot)):
        for j in range(len(one_hot[i])):
            if one_hot[j][i] != 0 and one_hot[j][i] != 1:
                return None
            if one_hot[j][i] == 1:
                mx[i] = j
    if mx.max()+1 != one_hot.shape[1]:
        return None
    return mx
