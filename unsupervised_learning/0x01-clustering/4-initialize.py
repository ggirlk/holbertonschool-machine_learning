#!/usr/bin/env python3
""" performs K-means on a dataset """

import numpy as np

kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """ doc """
    if type(X) is not np.ndarray or X.ndim != 2\
       or type(k) is not int or k < 1:
        return None, None, None
    n, d = X.shape
    
    pi = np.full((k,), 1/k)
    m, _ = kmeans(X, k)
    S = np.full((k, d, d), np.identity(d))
    return pi, m, S
