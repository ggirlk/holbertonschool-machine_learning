#!/usr/bin/env python3
""" performs K-means on a dataset """

import scipy.cluster.hierarchy as scp
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """ doc """
    dendrogram, linkage = scp.dendrogram, scp.linkage
    n, d = X.shape

    linked = linkage(X, 'ward')

    labelList = range(1, n+1)

    plt.figure(figsize=(10, 7))

    r = dendrogram(linked,
                   p=dist,
                   orientation='top',
                   labels=labelList,
                   color_threshold=55,
                   distance_sort='descending',
                   show_leaf_counts=True)
    
    plt.show()
    return scp.fcluster(linked, t=dist, criterion='distance')
