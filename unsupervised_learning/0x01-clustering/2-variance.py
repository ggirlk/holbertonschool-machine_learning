#!/usr/bin/env python3
""" performs K-means on a dataset """

import numpy as np


def variance(X, C):
    """ doc """
    try:
        sub = np.apply_along_axis(np.subtract, 1, X, C)
        return ((sub)**2).sum(axis=2).min(axis=1).sum()
    except Exception:
        return None
