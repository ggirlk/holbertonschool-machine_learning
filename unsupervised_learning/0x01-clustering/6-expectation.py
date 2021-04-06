#!/usr/bin/env python3
""" performs K-means on a dataset """

import numpy as np

pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """ doc """
    if type(X) is not np.ndarray or X.ndim != 2\
       or type(pi) is not np.ndarray or pi.ndim != 1\
       or type(m) is not np.ndarray or m.ndim != 2\
       or type(S) is not np.ndarray or S.ndim != 3:
        return None, None
    try:
        n, d = X.shape
        k = pi.shape[0]
        k1, d1 = m.shape
        k2, d2, d3 = S.shape
        if (k != k1 or k != k2 or k1 != k2)\
           or (d != d1 or d != d2 or d != d3)\
           or (d1 != d2 or d1 != d3 or d2 != d3)\
           or np.any(np.linalg.det(S)) == 0\
           or not np.isclose(pi.sum(), 1):
            return None, None
        pdfs = np.ndarray((k, n))
        for i in range(k):
            pdfs[i] = pdf(X, m[i], S[i])
        pdfs = pdfs * pi[:, np.newaxis]
        pdfsum = pdfs.sum(axis=0)
        # posterior probabilities for each data point in each cluster
        g = pdfs / pdfsum
        total_log_likelihood = np.log(pdfsum).sum()
        return g, total_log_likelihood
    except Exception:
        return None, None
