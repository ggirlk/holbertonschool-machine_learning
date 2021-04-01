#!/usr/bin/env python3
""" initialize cluster centroids for K-means """

import numpy as np


def initialize(X, k):
    """ doc """
    try:
        n, d = X.shape
        if k == 0 or d == 0:
            return None
        return np.random.uniform(np.amin(X, axis=0),
                                 np.amax(X, axis=0),
                                 size=(k, d))
    except Exception:
        return None
