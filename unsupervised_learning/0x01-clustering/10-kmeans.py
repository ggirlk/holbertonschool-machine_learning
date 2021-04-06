#!/usr/bin/env python3
""" performs K-means on a dataset """

import sklearn.cluster as cl


def kmeans(X, k, iterations=1000):
    """ doc """
    n, d = X.shape
    km = cl.KMeans(n_clusters=k, n_init=1, max_iter=iterations).fit(X)
    return km.cluster_centers_, km.labels_
