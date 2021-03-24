#!/usr/bin/env python3
""" PCA """
import numpy as np


def pca(X, var=0.95):
    """ performs PCA on a dataset """
    _, s, vh = np.linalg.svd(X)
    n = 0
    s_var = s.sum() * var
    tot = s[0]
    while(tot < s_var):
        n += 1
        tot += s[n]
    return vh[:n+1].T
