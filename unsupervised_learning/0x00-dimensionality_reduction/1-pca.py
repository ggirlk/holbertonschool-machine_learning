#!/usr/bin/env python3
""" PCA """


def pca(X, ndim):
    """ performs PCA on a dataset """
    u, s, vh = np.linalg.svd(X)

    return vh[:ndim].T
