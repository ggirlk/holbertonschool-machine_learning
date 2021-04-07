#!/usr/bin/env python3
""" performs K-means on a dataset """

import sklearn.mixture as gm


def gmm(X, k):
    """ doc """
    gmm = gm.GaussianMixture(n_components=k).fit(X)
    pi = gmm.weights_
    m = gmm.means_
    S = gmm.covariances_
    clss = gmm.predict(X)
    bic = gmm.bic(X)
    return pi, m, S, clss, bic
