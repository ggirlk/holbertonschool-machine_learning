#!/usr/bin/env python3
""" performs K-means on a dataset """
import numpy as np


def maximization(X, g):
    """ doc """
    if ((type(X) is not np.ndarray or type(g) is not np.ndarray
         or X.ndim != 2 or g.ndim != 2 or X.shape[0] != g.shape[1]
         or not np.all(np.isclose(g.sum(axis=0), 1)))):
        return None, None, None
    try:
        gsum = g.sum(axis=1)
        pi = gsum / X.shape[0]
        m = np.matmul(g, X) / gsum[:, np.newaxis]
        S = np.ndarray((m.shape[0], m.shape[1], m.shape[1]))
        for i in range(g.shape[0]):
            X_m = X - m[i]
            S[i] = (np.matmul((X_m * g[i, :, np.newaxis]).T, X_m) / gsum[i])
        return pi, m, S
    except Exception:
        return None, None, None
