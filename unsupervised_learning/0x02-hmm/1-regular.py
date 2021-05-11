#!/usr/bin/env python3
""" Hidden Markov Models """
import numpy as np


def regular(P):
    """
    determines the steady state probabilities
    of a regular markov chain
    """
    if type(P) is not np.ndarray or P.ndim != 2\
       or P.shape[0] != P.shape[1] or np.any(P <= 0)\
       or not np.all(np.isclose(P.sum(axis=1), 1)):
        return None
    try:
        dim = P.shape[0]
        q = (P - np.eye(dim))
        ones = np.ones(dim)
        q = np.c_[q, ones]
        QTQ = np.dot(q, q.T)
        bQT = np.ones(dim)
        return np.expand_dims(np.linalg.solve(QTQ, bQT), 0)
    except Exception:
        return None
