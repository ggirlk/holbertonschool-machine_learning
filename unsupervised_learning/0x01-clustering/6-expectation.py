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
        k, d = m.shape
        pdfs = np.ndarray((k, n))
        for i in range(k):
            pdfs[i] = pdf(X, m[i], S[i])
        pdfs = pdfs * pi[:, np.newaxis]
        pdfsum = pdfs.sum(axis=0)
        g = pdfs / pdfsum
        l = np.log(pdfsum).sum()
        return g, l
    except Exception:
        return None, None
