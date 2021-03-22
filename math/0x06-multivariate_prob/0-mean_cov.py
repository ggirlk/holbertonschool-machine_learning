#!/usr/bin/env python3
""" mean cov """
import numpy as np


def mean_cov(X):
    """
    calculate the mean and
    covariance of a data set
    """
    mean = X.mean(axis=0, keepdims=True)
    n = X.shape[0] - 1
    x = X - mean
    cov = np.dot(x.T, X.conj()) / n
    return mean, cov
