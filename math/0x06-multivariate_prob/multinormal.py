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

    def pdf(self, x):
        """ calculate the PDF at a data point """
        if type(x) is not np.ndarray:
            raise TypeError("x must be a numpy.ndarray")
        if len(x.shape) < 2 or x.shape[1] != 1:
            raise ValueError("x must have the shape ({d}, 1)")
        d = x.shape[0]
        x_m = x - self.mean
        return (1 / (np.sqrt((2 * np.pi)**d * np.linalg.det(self.cov))) *
                np.exp(-(np.linalg.solve(self.cov, x_m).T.dot(x_m)) / 2))
