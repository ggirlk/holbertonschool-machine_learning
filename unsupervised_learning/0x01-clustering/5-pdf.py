#!/usr/bin/env python3
""" performs K-means on a dataset """

import numpy as np

kmeans = __import__('1-kmeans').kmeans


def pdf(X, m, S):
    """ doc """
    if type(X) is not np.ndarray or X.ndim != 2\
       or type(m) is not np.ndarray or m.ndim != 1\
       or type(S) is not np.ndarray or S.ndim != 2:
        return None
    try:
        n, d = X.shape
        det = np.linalg.det(S)
        if det == 0:
            return None
        X_m = (X - m).T
        Xmul = (X_m * np.matmul(np.linalg.inv(S), X_m)).sum(axis=0)
        exp = np.exp(Xmul / -2)
        sqrt = np.sqrt(np.power(2 * np.pi, d) * det)
        P = (exp / sqrt)
        P = np.maximum(P, 1e-300)
        return P
    except Exception:
        return None
