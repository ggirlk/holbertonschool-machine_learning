#!/usr/bin/env python3
"""
converts a numeric label
vector into a one-hot matrix
"""
import numpy as np


def one_hot_encode(Y, classes):
    mx = np.zeros((classes, classes))
    i = 0
    for n in Y:
        mx[n:n+1, i:i+1] = 1
        i += 1
    return mx
