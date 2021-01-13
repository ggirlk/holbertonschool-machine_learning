#!/usr/bin/env python3
"""
shuffles the data points
in two matrices the same way
"""
import numpy as np


def shuffle_data(X, Y):
    """ doc """
    x = np.random.permutation(X.shape[0])
    return X[x], Y[y]
