#!/usr/bin/env python3
""" corr """
import numpy as np


def correlation(C):
    """
    calculate a correlation matrix
    """
    if type(C) is not np.ndarray:
        raise TypeError("C must be a numpy.ndarray")
    if C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square")
    corr = np.ndarray((C.shape))
    d = C.shape[0]
    for i in range(d):
        for j in range(d):
            corr[i,j] = C[i,j]/(np.sqrt(C[i,i]*C[j,j]))
    return corr
