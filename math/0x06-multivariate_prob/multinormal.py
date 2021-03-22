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
    cov = np.dot(x.T, X.conj()) / n
    return mean, cov


class MultiNormal():
    """ Multivariate Normal distribution """
    
    def __init__(self, data):
        """ constructor """
        if type(data) is not np.ndarray or len(data.shape) < 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = data.mean(axis=1, keepdims=True)
        _, self.cov = mean_cov(data.T)
