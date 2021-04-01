#!/usr/bin/env python3
""" performs K-means on a dataset """

import numpy as np
from sklearn.cluster import KMeans


def kmeans(X, k, iterations=1000):
    """ doc """
    n, d = X.shape
    init = np.random.uniform(np.amin(X, axis=0),
                                 np.amax(X, axis=0),
                                 size=(k, d))
    km = KMeans(n_clusters=k, init=init, n_init=1, max_iter=iterations).fit(X)
    return km.cluster_centers_, km.labels_
