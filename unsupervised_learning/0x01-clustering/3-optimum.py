#!/usr/bin/env python3
""" performs K-means on a dataset """

import numpy as np

kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """ doc """
    if type(X) is not np.ndarray or X.ndim != 2\
       or type(C) is not np.ndarray or C.ndim != 2:
        return None

    try:
        sub = np.apply_along_axis(np.subtract, 1, X, C)
        return ((sub)**2).sum(axis=2).min(axis=1).sum()
    except Exception:
        return None
