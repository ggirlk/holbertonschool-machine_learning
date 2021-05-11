#!/usr/bin/env python3
""" Hidden Markov Models """
import numpy as np


def absorbing(P):
    """
    determines if a markov chain is absorbing
    """
    if type(P) is not np.ndarray or P.ndim != 2\
       or P.shape[0] != P.shape[1] or np.any(P < 0)\
       or not np.all(np.isclose(P.sum(axis=1), 1)):
        return None
    for i in range(P.shape[0]):
        if P[i, i] == 1:
            return True
    return False
