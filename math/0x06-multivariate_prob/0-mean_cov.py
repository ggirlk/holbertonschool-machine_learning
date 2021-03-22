#!/usr/bin/env python3
""" corr """
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
    cov = np.dot(x.T, X.conj()) / n
    return mean, cov

def correlation(C):
    """
    calculate a correlation matrix
    """
    if type(C) is not np.ndarray:
        raise TypeError("C must be a numpy.ndarray")
    if C.shape[0] != C.shape[0]:
        raise ValueError("C must be a 2D square")
    _, cov = mean_cov(C)
    return (np.corrcoef(C))
