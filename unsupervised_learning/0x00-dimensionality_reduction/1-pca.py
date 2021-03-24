#!/usr/bin/env python3
""" PCA """
import numpy as np


def pca(X, ndim):
    """ performs PCA on a dataset """
    X_m = X - X.mean(axis=0)
    _, s, vh = np.linalg.svd(X_m)
    return np.dot(X_m, vh[:ndim].T)
