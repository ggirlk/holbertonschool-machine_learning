#!/usr/bin/env python3
""" mean cov """
import numpy as np


def mean_cov(X):
    """
    calculate the mean and
    covariance of a data set
    """
    if type(X) != np.ndarray or (len(X.shape) != 2):
        raise TypeError("X must be a 2D numpy.ndarray")
    n = X.shape[0] - 1
    if n < 2:
        raise ValueError("X must contain multiple data points")
    mean = X.mean(axis=0, keepdims=True)
    x = X - mean
    cov = np.dot(x.T, x) / n
    return mean, cov
