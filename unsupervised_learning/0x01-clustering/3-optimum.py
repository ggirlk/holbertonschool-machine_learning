#!/usr/bin/env python3
""" performs K-means on a dataset """

import numpy as np

kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """ doc """
    if type(X) is not np.ndarray:
        return None, None
    if kmax is None:
        kmax = X.shape[0]
    if ((X.ndim != 2 or type(kmin) is not int
         or kmin < 1 or type(iterations) is not int or iterations < 1
         or type(kmax) is not int or kmax <= kmin)):
        return None, None
    results = [kmeans(X, kmin, iterations)]
    firstvar = variance(X, results[0][0])
    d_vars = [0]
    while kmin < kmax:
        C, clss = kmeans(X, kmin, iterations)
        vari = variance(X, C)
        results.append((C, clss))
        d_vars.append(firstvar - vari)
        kmin += 1
    return results, d_vars
